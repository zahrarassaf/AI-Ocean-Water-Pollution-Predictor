"""
Ocean Pollution Prediction System - Q1 Edition
==============================================

Machine learning pipeline for predicting ocean pollution levels
from satellite-derived NetCDF datasets.

Pipeline stages:
    1. Load NetCDF files (xarray)
    2. Sample 3-D points (time x lat x lon)
    3. Extract features across datasets with spatial-temporal imputation
    4. Build a continuous pollution score -> discretize into LOW/MEDIUM/HIGH
    5. Train a gradient-boosted / random-forest classifier with CV
    6. Persist the model + metadata + feature statistics
    7. Predict from partial inputs using calibrated fallbacks

Author : Ocean Pollution Project
Version: 2.1.0
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time
import traceback
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("ocean.predictor")

VERSION = "2.1.0"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    """All tuneable hyper-parameters in one place."""

    data_path: str = "data/raw"
    model_dir: str = "models"

    # Sampling
    sample_limit: int = 50_000
    min_valid_ratio: float = 0.30
    random_state: int = 42

    # Feature engineering
    exclude_vars: Tuple[str, ...] = (
        "time", "latitude", "longitude", "lat", "lon",
    )

    # Labelling
    #   method: "percentile" | "absolute"
    label_method: str = "percentile"
    low_percentile: float = 33.0
    medium_percentile: float = 66.0
    #   Used when method == "absolute" (pollution score in [0,1])
    low_absolute: float = 0.33
    high_absolute: float = 0.66

    # Pollution score weights (renormalised over available indicators)
    indicator_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "CHL": 0.30,
            "KD490": 0.25,
            "PP": 0.20,
            "CDM": 0.15,
            "BBP": 0.10,
        }
    )

    # Model selection
    model_type: str = "random_forest"  # "random_forest" | "hist_gb"
    test_size: float = 0.20
    cv_folds: int = 5
    n_estimators: int = 200
    max_depth: int = 15

    # Persist
    persist: bool = True


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def _model_paths(model_dir: str) -> Dict[str, Path]:
    base = Path(model_dir)
    return {
        "base": base,
        "model": base / "ocean_model.pkl",
        "scaler": base / "ocean_scaler.pkl",
        "features": base / "features.txt",
        "metadata": base / "metadata.json",
        "stats": base / "feature_statistics.json",
    }


def _remove_duplicates(items: Iterable[str]) -> List[str]:
    """Preserve order while removing duplicates."""
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
class NetCDFLoader:
    """Open and manage a directory of NetCDF files."""

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.datasets: Dict[str, xr.Dataset] = {}

    def load(self) -> bool:
        if not self.data_path.exists():
            logger.error("Data path does not exist: %s", self.data_path)
            return False

        files = sorted(glob.glob(str(self.data_path / "*.nc")))
        if not files:
            logger.error("No .nc files found in %s", self.data_path)
            return False

        logger.info("Found %d NetCDF files", len(files))
        for fp in files:
            name = os.path.basename(fp)
            try:
                ds = xr.open_dataset(fp)
                self.datasets[name] = ds
                logger.info("  Loaded %-40s (%d vars)", name, len(ds.data_vars))
            except Exception as exc:  # noqa: BLE001
                logger.warning("  Failed to load %s: %s", name, exc)

        for name, ds in self.datasets.items():
            logger.debug("%s -> %s", name, list(ds.data_vars))

        return bool(self.datasets)

    def find_chlorophyll(self) -> Tuple[Optional[xr.Dataset], Optional[str], Optional[str]]:
        for name, ds in self.datasets.items():
            for var in ds.data_vars:
                upper = var.upper()
                if any(k in upper for k in ("CHL", "CHLOROPHYLL")):
                    logger.info("Chlorophyll found: %s / %s", name, var)
                    return ds, name, var
        logger.warning("No chlorophyll variable found across datasets")
        return None, None, None

    def close_all(self) -> None:
        for ds in self.datasets.values():
            try:
                ds.close()
            except Exception:  # noqa: BLE001
                pass
        self.datasets.clear()

    def __enter__(self) -> "NetCDFLoader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close_all()


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
class FeatureExtractor:
    """Extract features at sampled 3-D points with median imputation."""

    def __init__(
        self,
        datasets: Dict[str, xr.Dataset],
        exclude_vars: Sequence[str],
        min_valid_ratio: float,
    ):
        self.datasets = datasets
        self.exclude = {v.lower() for v in exclude_vars}
        self.min_valid_ratio = min_valid_ratio

    def _all_vars(self) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        for name, ds in self.datasets.items():
            for var in ds.data_vars:
                if var.lower() in self.exclude:
                    continue
                out.append((name, var))
        return out

    def extract(self, points: List[Dict[str, int]]) -> Tuple[Dict[str, np.ndarray], List[str]]:
        feature_data: Dict[str, np.ndarray] = {}
        feature_names: List[str] = []

        all_vars = self._all_vars()
        logger.info("Total candidate features: %d", len(all_vars))

        for fname, var in all_vars:
            if var in feature_data:
                continue
            try:
                arr = self.datasets[fname][var].values
                if arr.ndim != 3:
                    logger.debug("  Skip %s: ndim=%d", var, arr.ndim)
                    continue

                values: List[float] = []
                valid = 0
                for p in points:
                    try:
                        v = float(arr[p["time_idx"], p["lat_idx"], p["lon_idx"]])
                        if np.isfinite(v):
                            values.append(v)
                            valid += 1
                        else:
                            values.append(np.nan)
                    except (IndexError, ValueError, TypeError):
                        values.append(np.nan)

                ratio = valid / max(len(points), 1)
                if ratio < self.min_valid_ratio:
                    logger.debug("  Skip %s: valid=%.1f%%", var, ratio * 100)
                    continue

                arr_np = np.asarray(values, dtype=float)
                if valid > 0:
                    arr_np[np.isnan(arr_np)] = np.nanmedian(arr_np)
                else:
                    continue
                feature_data[var] = arr_np
                feature_names.append(var)
                logger.info("  Kept %-20s (valid=%.1f%%)", var, ratio * 100)
            except Exception as exc:  # noqa: BLE001
                logger.warning("  Error on %s: %s", var, exc)
                continue

        return feature_data, feature_names


# ---------------------------------------------------------------------------
# Pollution score & labelling
# ---------------------------------------------------------------------------
class PollutionLabeler:
    """Compute a continuous pollution score and discretize it."""

    def __init__(self, config: TrainingConfig):
        self.cfg = config

    def _resolve_indicators(self, feature_names: Sequence[str]) -> Dict[str, float]:
        """Match configured indicators to actual feature names."""
        resolved: Dict[str, float] = {}
        for target, weight in self.cfg.indicator_weights.items():
            # exact
            if target in feature_names:
                resolved[target] = weight
                continue
            # case-insensitive contains
            for feat in feature_names:
                if target.upper() in feat.upper():
                    resolved[feat] = weight
                    break
        if not resolved:
            # fallback: equal weights
            n = len(feature_names)
            resolved = {f: 1.0 / n for f in feature_names}
        return resolved

    def score(self, X: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
        indicators = self._resolve_indicators(feature_names)
        total_weight = sum(indicators.values())

        score = np.zeros(X.shape[0], dtype=float)
        for feat, weight in indicators.items():
            if feat not in feature_names:
                continue
            idx = feature_names.index(feat)
            col = X[:, idx]
            lo, hi = np.nanmin(col), np.nanmax(col)
            if hi <= lo:
                continue
            normalized = (col - lo) / (hi - lo)
            score += normalized * (weight / total_weight)
        return score

    def labels(self, score: np.ndarray) -> np.ndarray:
        if self.cfg.label_method == "percentile":
            low_t = np.percentile(score, self.cfg.low_percentile)
            high_t = np.percentile(score, self.cfg.medium_percentile)
        elif self.cfg.label_method == "absolute":
            low_t = self.cfg.low_absolute
            high_t = self.cfg.high_absolute
        else:
            raise ValueError(f"Unknown label method: {self.cfg.label_method}")

        y = np.zeros(len(score), dtype=int)
        y[(score > low_t) & (score <= high_t)] = 1
        y[score > high_t] = 2
        logger.info(
            "Labelling thresholds: LOW<=%.3f<MEDIUM<=%.3f<HIGH",
            low_t,
            high_t,
        )
        return y


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class ModelTrainer:
    """Train, evaluate and persist the classifier."""

    def __init__(self, config: TrainingConfig):
        self.cfg = config
        self.model = None
        self.scaler: Optional[StandardScaler] = None
        self.features: List[str] = []
        self.feature_statistics: Dict[str, Dict[str, float]] = {}
        self.metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    def _build_dataset(
        self,
        loader: NetCDFLoader,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[str]]]:
        chl_ds, chl_file, chl_var = loader.find_chlorophyll()
        if chl_ds is None:
            if not loader.datasets:
                return None, None, None
            chl_file = next(iter(loader.datasets))
            chl_ds = loader.datasets[chl_file]
            chl_var = next(iter(chl_ds.data_vars))
            logger.warning("Falling back to reference %s / %s", chl_file, chl_var)

        ref = chl_ds[chl_var].values
        if ref.ndim != 3:
            logger.error("Reference variable %s is not 3-D", chl_var)
            return None, None, None

        flat = ref.flatten()
        valid_idx = np.where(np.isfinite(flat))[0]
        if valid_idx.size == 0:
            logger.error("No valid reference points")
            return None, None, None

        rng = np.random.default_rng(self.cfg.random_state)
        if valid_idx.size > self.cfg.sample_limit:
            chosen = rng.choice(valid_idx, self.cfg.sample_limit, replace=False)
        else:
            chosen = valid_idx
        logger.info("Sampled %d points", len(chosen))

        t_size, lat_size, lon_size = ref.shape
        points = []
        for idx in chosen:
            t = int(idx // (lat_size * lon_size))
            rem = int(idx % (lat_size * lon_size))
            la = int(rem // lon_size)
            lo = int(rem % lon_size)
            points.append(
                {"time_idx": t, "lat_idx": la, "lon_idx": lo, "ref": float(flat[idx])}
            )

        extractor = FeatureExtractor(
            loader.datasets, self.cfg.exclude_vars, self.cfg.min_valid_ratio
        )
        feature_data, feature_names = extractor.extract(points)
        if not feature_names:
            logger.error("No features extracted")
            return None, None, None

        X = np.column_stack([feature_data[name] for name in feature_names])
        self.features = _remove_duplicates(feature_names)

        # statistics per feature
        for i, name in enumerate(feature_names):
            col = X[:, i]
            self.feature_statistics[name] = {
                "mean": float(np.nanmean(col)),
                "std": float(np.nanstd(col) or 1.0),
                "min": float(np.nanmin(col)),
                "max": float(np.nanmax(col)),
                "median": float(np.nanmedian(col)),
            }

        labeler = PollutionLabeler(self.cfg)
        score = labeler.score(X, feature_names)
        y = labeler.labels(score)

        logger.info("Class distribution:")
        for cls_id, cls_name in enumerate(("LOW", "MEDIUM", "HIGH")):
            count = int(np.sum(y == cls_id))
            logger.info("  %-7s : %7d (%.1f%%)", cls_name, count, count / len(y) * 100)

        return X, y, self.features

    # ------------------------------------------------------------------
    def _build_classifier(self) -> Any:
        if self.cfg.model_type == "hist_gb":
            return HistGradientBoostingClassifier(
                max_iter=self.cfg.n_estimators,
                max_depth=self.cfg.max_depth,
                learning_rate=0.05,
                random_state=self.cfg.random_state,
            )
        return RandomForestClassifier(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.cfg.random_state,
            n_jobs=-1,
            class_weight="balanced",
            oob_score=True,
            max_features="sqrt",
        )

    # ------------------------------------------------------------------
    def train(self) -> bool:
        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE STARTED  (v%s)", VERSION)
        logger.info("=" * 60)

        with NetCDFLoader(self.cfg.data_path) as loader:
            if not loader.load():
                return False

            try:
                X, y, feature_names = self._build_dataset(loader)
            finally:
                loader.close_all()

        if X is None or y is None or feature_names is None:
            logger.error("Dataset construction failed")
            return False

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
            stratify=y,
        )
        logger.info(
            "Split -> train: %d | test: %d", len(X_train), len(X_test)
        )

        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        self.model = self._build_classifier()
        logger.info("Fitting %s ...", type(self.model).__name__)
        t0 = time.time()
        self.model.fit(X_train_s, y_train)
        logger.info("Training finished in %.1fs", time.time() - t0)

        # Evaluation
        y_train_pred = self.model.predict(X_train_s)
        y_test_pred = self.model.predict(X_test_s)
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        cv = StratifiedKFold(
            n_splits=self.cfg.cv_folds,
            shuffle=True,
            random_state=self.cfg.random_state,
        )
        cv_scores = cross_val_score(self.model, X_train_s, y_train, cv=cv, n_jobs=-1)

        logger.info("Training accuracy : %.4f", train_acc)
        logger.info("Test accuracy     : %.4f", test_acc)
        logger.info("CV mean           : %.4f (+/- %.4f)", cv_scores.mean(), cv_scores.std())
        if hasattr(self.model, "oob_score_"):
            logger.info("OOB score         : %.4f", self.model.oob_score_)

        logger.info("\n%s", classification_report(y_test, y_test_pred, target_names=["LOW", "MEDIUM", "HIGH"]))

        cm = confusion_matrix(y_test, y_test_pred)
        logger.info("Confusion matrix:\n%s", cm)

        if train_acc - test_acc > 0.05:
            logger.warning(
                "Possible overfitting (train-test gap = %.4f)",
                train_acc - test_acc,
            )

        # Feature importance (only for tree models exposing it)
        importances = getattr(self.model, "feature_importances_", None)
        top_features: List[Tuple[str, float]] = []
        if importances is not None:
            order = np.argsort(importances)[::-1]
            for idx in order[:10]:
                top_features.append((self.features[idx], float(importances[idx])))
                logger.info("  %-25s : %.4f", self.features[idx], importances[idx])

        self.metadata = {
            "version": VERSION,
            "training_date": datetime.now().isoformat(),
            "model_type": type(self.model).__name__,
            "features": self.features,
            "feature_count": len(self.features),
            "samples": int(X.shape[0]),
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "classes": ["LOW", "MEDIUM", "HIGH"],
            "confusion_matrix": cm.tolist(),
            "top_features": top_features,
            "config": asdict(self.cfg),
        }

        if self.cfg.persist:
            self._persist()

        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE COMPLETED")
        logger.info("=" * 60)
        return True

    # ------------------------------------------------------------------
    def _persist(self) -> None:
        paths = _model_paths(self.cfg.model_dir)
        paths["base"].mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, paths["model"])
        joblib.dump(self.scaler, paths["scaler"])

        paths["features"].write_text("\n".join(self.features), encoding="utf-8")

        paths["metadata"].write_text(
            json.dumps(self.metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        paths["stats"].write_text(
            json.dumps(self.feature_statistics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Artifacts saved to %s/", paths["base"])


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------
class OceanPollutionPredictor:
    """Inference wrapper usable by the API layer and CLI."""

    CLASSES = ["LOW", "MEDIUM", "HIGH"]

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.cfg = config or TrainingConfig()
        self.model = None
        self.scaler: Optional[StandardScaler] = None
        self.features: List[str] = []
        self.feature_statistics: Dict[str, Dict[str, float]] = {}
        self.metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    def load(self) -> bool:
        paths = _model_paths(self.cfg.model_dir)
        if not paths["model"].exists():
            logger.error("Model file not found: %s", paths["model"])
            return False
        try:
            self.model = joblib.load(paths["model"])
            self.scaler = joblib.load(paths["scaler"])
            self.features = _remove_duplicates(
                paths["features"].read_text(encoding="utf-8").splitlines()
            )
            if paths["stats"].exists():
                self.feature_statistics = json.loads(
                    paths["stats"].read_text(encoding="utf-8")
                )
            if paths["metadata"].exists():
                self.metadata = json.loads(
                    paths["metadata"].read_text(encoding="utf-8")
                )
            logger.info(
                "Model loaded: %d features | acc=%s",
                len(self.features),
                self.metadata.get("test_accuracy", "N/A"),
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to load model: %s", exc)
            return False

    # ------------------------------------------------------------------
    def _fallback_from_chl(self, feature: str, chl: float) -> float:
        upper = feature.upper()
        if "CHL" in upper and "UNCERTAINTY" not in upper:
            return chl
        if "CHL" in upper and "UNCERTAINTY" in upper:
            return chl * 0.1
        if "PP" in upper:
            return chl * 100.0
        if "KD" in upper:
            return 0.1 + chl * 0.05
        if any(
            k in upper
            for k in (
                "DIATO", "DINO", "GREEN", "HAPTO",
                "MICRO", "NANO", "PICO", "PROCHLO", "PROKAR",
            )
        ):
            return chl * 0.005 if "UNCERTAINTY" in upper else chl * 0.01
        if "CDM" in upper:
            return 0.01 + chl * 0.002
        if "BBP" in upper:
            return 0.001 + chl * 0.0003
        return 0.0

    def _prepare_row(self, payload: Dict[str, Any]) -> Optional[Dict[str, float]]:
        # find CHL
        chl: Optional[float] = None
        for k, v in payload.items():
            if "CHL" in k.upper() and "UNCERTAINTY" not in k.upper():
                try:
                    chl = float(v)
                    break
                except (TypeError, ValueError):
                    continue
        if chl is None:
            return None

        prepared: Dict[str, float] = {}
        for feat in self.features:
            # exact match
            matched = False
            for k, v in payload.items():
                if k.upper() == feat.upper():
                    try:
                        prepared[feat] = float(v)
                        matched = True
                        break
                    except (TypeError, ValueError):
                        continue
            if matched:
                continue
            # fallback to median from training statistics, then CHL heuristics
            if feat in self.feature_statistics:
                prepared[feat] = float(self.feature_statistics[feat]["median"])
            else:
                prepared[feat] = self._fallback_from_chl(feat, chl)
        return prepared

    # ------------------------------------------------------------------
    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None or self.scaler is None:
            return {"error": "Model is not loaded"}

        try:
            row = self._prepare_row(payload)
            if row is None:
                return {"error": "CHL value is required"}

            df = pd.DataFrame([row], columns=self.features)
            scaled = self.scaler.transform(df)

            pred_idx = int(self.model.predict(scaled)[0])
            proba = self.model.predict_proba(scaled)[0]

            return {
                "level": pred_idx,
                "level_name": self.CLASSES[pred_idx],
                "confidence": float(np.max(proba)),
                "probabilities": {
                    self.CLASSES[i]: float(proba[i]) for i in range(len(self.CLASSES))
                },
                "chl_value": float(row.get(self.features[0], 0.0)),
                "features_provided": len(payload),
                "features_total": len(self.features),
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("Prediction failed: %s", exc)
            return {"error": str(exc), "traceback": traceback.format_exc()}

    # ------------------------------------------------------------------
    def predict_batch(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.predict(row) for row in rows]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli_train(cfg: TrainingConfig) -> int:
    trainer = ModelTrainer(cfg)
    return 0 if trainer.train() else 1


def _cli_demo(predictor: OceanPollutionPredictor) -> int:
    cases = [
        ("Open Ocean", {"CHL": 0.3}),
        ("Coastal Bay", {"CHL": 2.5, "PP": 400, "KD490": 0.2}),
        ("Polluted Estuary", {"CHL": 8.0, "PP": 800, "KD490": 0.5, "DIATO": 0.2}),
        ("Clear Coastal", {"CHL": 0.8}),
        ("Algal Bloom", {"CHL": 15.0, "PP": 1200, "GREEN": 0.5}),
    ]
    for name, data in cases:
        result = predictor.predict(data)
        if "error" in result:
            logger.error("%-20s -> %s", name, result["error"])
            continue
        logger.info(
            "%-20s -> %-7s (%.1f%%)",
            name,
            result["level_name"],
            result["confidence"] * 100,
        )
    return 0


def _cli_interactive(predictor: OceanPollutionPredictor) -> int:
    logger.info("Interactive mode. Type 'quit' to exit.")
    while True:
        try:
            raw = input("CHL (mg/m3): ").strip()
            if raw.lower() in {"quit", "exit"}:
                break
            chl = float(raw)
        except ValueError:
            logger.warning("Invalid number, try again.")
            continue

        extra_raw = input("Additional features (KEY=VAL,KEY=VAL) or empty: ").strip()
        payload: Dict[str, float] = {"CHL": chl}
        if extra_raw:
            for chunk in extra_raw.split(","):
                if "=" not in chunk:
                    continue
                k, v = chunk.split("=", 1)
                try:
                    payload[k.strip().upper()] = float(v.strip())
                except ValueError:
                    logger.warning("Skipping %s", chunk)

        result = predictor.predict(payload)
        if "error" in result:
            logger.error(result["error"])
            continue
        logger.info(
            "Result: %s (%.1f%%)",
            result["level_name"],
            result["confidence"] * 100,
        )
        for level, prob in result["probabilities"].items():
            logger.info("  %-7s : %.1f%%", level, prob * 100)
    return 0


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ocean Pollution Prediction System (Q1)",
    )
    p.add_argument("-t", "--train", action="store_true", help="Train a new model")
    p.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")
    p.add_argument("-d", "--demo", action="store_true", help="Run demo predictions")
    p.add_argument("-p", "--predict", nargs="*", help="Quick predict: CHL=2.5 PP=300")
    p.add_argument("--data-path", default=None, help="Override data_path")
    p.add_argument("--model-dir", default=None, help="Override model_dir")
    p.add_argument("--model-type", choices=["random_forest", "hist_gb"], default=None)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)

    cfg = TrainingConfig()
    if args.data_path:
        cfg.data_path = args.data_path
    if args.model_dir:
        cfg.model_dir = args.model_dir
    if args.model_type:
        cfg.model_type = args.model_type

    if args.train:
        return _cli_train(cfg)

    predictor = OceanPollutionPredictor(cfg)
    if not predictor.load():
        logger.error("No model available. Train one first with --train")
        return 1

    if args.demo:
        return _cli_demo(predictor)

    if args.interactive:
        return _cli_interactive(predictor)

    if args.predict is not None:
        payload: Dict[str, float] = {}
        for token in args.predict:
            if "=" in token:
                k, v = token.split("=", 1)
                try:
                    payload[k.strip().upper()] = float(v)
                except ValueError:
                    logger.warning("Ignoring %s", token)
        if not payload:
            logger.error("No features provided. Example: -p CHL=2.5 PP=300")
            return 1
        result = predictor.predict(payload)
        if "error" in result:
            logger.error(result["error"])
            return 1
        logger.info(
            "Result: %s (%.1f%%)",
            result["level_name"],
            result["confidence"] * 100,
        )
        return 0

    _build_argparser().print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
