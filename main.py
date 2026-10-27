# main.py
# Streamlit Dashboard for Ocean Pollution Prediction (v2.1.0)
# Run with: streamlit run main.py

from __future__ import annotations

import logging
import os
import platform
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("ocean-dashboard")

APP_VERSION = "2.1.0"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class APIConfig:
    """Immutable API configuration (env-overridable)."""

    BASE_URL: str = os.getenv("OCEAN_API_URL", "http://localhost:8000")
    TIMEOUT: int = int(os.getenv("OCEAN_API_TIMEOUT", "10"))
    MAX_RETRIES: int = int(os.getenv("OCEAN_API_RETRIES", "3"))

    POLLUTION_LEVELS: tuple[str, ...] = ("LOW", "MEDIUM", "HIGH")
    LEVEL_COLORS: Dict[str, str] = field(
        default_factory=lambda: {
            "LOW": "#4CAF50",
            "MEDIUM": "#FF9800",
            "HIGH": "#F44336",
        }
    )


API = APIConfig()


# ---------------------------------------------------------------------------
# HTTP session with retry
# ---------------------------------------------------------------------------
def _build_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=API.MAX_RETRIES,
        backoff_factor=0.3,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


_SESSION = _build_session()


# ---------------------------------------------------------------------------
# API Client
# ---------------------------------------------------------------------------
class APIClient:
    """Thin wrapper around the Ocean Pollution REST API."""

    @staticmethod
    def _get(path: str, timeout: Optional[int] = None) -> Optional[dict]:
        try:
            r = _SESSION.get(
                f"{API.BASE_URL}{path}", timeout=timeout or API.TIMEOUT
            )
            return r.json() if r.ok else None
        except requests.RequestException as e:
            logger.warning("GET %s failed: %s", path, e)
            return None

    @staticmethod
    def _post(path: str, payload: dict, timeout: Optional[int] = None) -> Optional[dict]:
        try:
            r = _SESSION.post(
                f"{API.BASE_URL}{path}",
                json=payload,
                timeout=timeout or API.TIMEOUT,
            )
            return r.json() if r.ok else None
        except requests.RequestException as e:
            logger.warning("POST %s failed: %s", path, e)
            return {"error": str(e)}

    # --- Cached endpoints ------------------------------------------------
    @staticmethod
    @st.cache_data(ttl=30, show_spinner=False)
    def get_health() -> Optional[dict]:
        return APIClient._get("/health", timeout=5)

    @staticmethod
    @st.cache_data(ttl=300, show_spinner=False)
    def get_model_info() -> Optional[dict]:
        return APIClient._get("/model-info")

    @staticmethod
    def predict_single(data: Dict[str, Any]) -> Optional[dict]:
        return APIClient._post("/predict", data)

    @staticmethod
    def predict_batch(data: Dict[str, Any]) -> Optional[dict]:
        return APIClient._post("/predict/batch", data, timeout=API.TIMEOUT * 2)

    @staticmethod
    def get_project_structure() -> Optional[dict]:
        return APIClient._get("/project/structure")

    @staticmethod
    def get_sample_prediction() -> Optional[dict]:
        return APIClient._get("/predict/sample")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
@dataclass
class AppState:
    predictions_history: List[dict] = field(default_factory=list)
    current_tab: str = "Dashboard"
    sample_data: Dict[str, float] = field(
        default_factory=lambda: {
            "CHL": 2.5,
            "PP": 350.0,
            "KD490": 0.25,
            "DIATO": 0.03,
            "DINO": 0.02,
        }
    )


def init_session_state() -> None:
    if "app_state" not in st.session_state:
        st.session_state.app_state = AppState()
    defaults = {
        "current_tab": "Dashboard",
        "predictions_history": [],
        "sample_data": {"CHL": 2.5, "PP": 350.0, "KD490": 0.25, "DIATO": 0.03, "DINO": 0.02},
        "run_sample": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def get_state() -> AppState:
    return st.session_state.app_state


# ---------------------------------------------------------------------------
# Reusable UI components
# ---------------------------------------------------------------------------
def section_title(text: str, level: int = 1) -> None:
    cls = "main-header" if level == 1 else "sub-header"
    tag = "h1" if level == 1 else "h3"
    st.markdown(f'<{tag} class="{cls}">{text}</{tag}>', unsafe_allow_html=True)


def metric_card(label: str, value: str) -> None:
    st.markdown(
        f'<div class="metric-card">{label}<br>'
        f'<span style="font-size:1.5rem;font-weight:bold;">{value}</span></div>',
        unsafe_allow_html=True,
    )


def prediction_card(level: str, confidence: float) -> None:
    st.markdown(
        f"""
        <div class="prediction-card {level.lower()}-pollution">
            <h3 style="text-align:center;margin:0;">{level}</h3>
            <p style="text-align:center;font-size:0.9rem;">
                Confidence: {confidence:.1%}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def probability_charts(probabilities: Dict[str, float], prefix: str = "rt") -> None:
    """Render bar + pie probability charts side-by-side."""
    colors = [API.LEVEL_COLORS.get(k, "#999") for k in probabilities]

    fig_bar = go.Figure(
        go.Bar(
            x=list(probabilities.keys()),
            y=list(probabilities.values()),
            marker_color=colors,
            text=[f"{v:.1%}" for v in probabilities.values()],
            textposition="auto",
            hovertemplate="<b>%{x}</b><br>Probability: %{y:.2%}<extra></extra>",
        )
    )
    fig_bar.update_layout(
        title="Pollution Level Probabilities",
        yaxis_title="Probability",
        yaxis_range=[0, 1],
        height=350,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    fig_pie = px.pie(
        values=list(probabilities.values()),
        names=list(probabilities.keys()),
        title="Probability Distribution",
        color=list(probabilities.keys()),
        color_discrete_map=API.LEVEL_COLORS,
        hole=0.3,
    )
    fig_pie.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Probability: %{percent}<extra></extra>",
    )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_bar, use_container_width=True, key=f"{prefix}_bar_{time.time_ns()}")
    with c2:
        st.plotly_chart(fig_pie, use_container_width=True, key=f"{prefix}_pie_{time.time_ns()}")


def recommendations(level: str) -> None:
    """Render recommendations based on pollution level."""
    if level == "LOW":
        st.success(
            "WATER QUALITY: EXCELLENT\n\n"
            "Recommended Actions:\n"
            "- Continue normal monitoring schedule\n"
            "- Maintain current conservation practices\n"
            "- Schedule next routine check in 4 weeks"
        )
    elif level == "MEDIUM":
        st.warning(
            "WATER QUALITY: MODERATE CONCERN\n\n"
            "Recommended Actions:\n"
            "- Increase monitoring frequency to twice weekly\n"
            "- Investigate potential pollution sources\n"
            "- Notify local environmental authorities"
        )
    else:  # HIGH
        st.error(
            "WATER QUALITY: HIGH ALERT\n\n"
            "Emergency Actions Required:\n"
            "1. IMMEDIATELY notify environmental authorities\n"
            "2. Activate emergency response plan\n"
            "3. Issue public health advisory\n"
            "4. Restrict water activities\n"
            "5. Deploy cleanup operations"
        )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
NAV_OPTIONS = [
    "Dashboard",
    "Real-time Prediction",
    "Batch Analysis",
    "Data Insights",
    "Model Information",
    "Prediction History",
    "Settings",
]


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            '<h2 style="text-align:center;">Ocean Pollution Monitor</h2>',
            unsafe_allow_html=True,
        )
        st.markdown("---")

        st.markdown("### Navigation")
        selected = st.radio(
            "Select Section:", NAV_OPTIONS, label_visibility="collapsed"
        )
        st.session_state.current_tab = selected

        st.markdown("---")
        st.markdown("### System Status")

        health = APIClient.get_health()
        if health:
            status = health.get("status", "unknown")
            model_loaded = health.get("model_loaded", False)
            c1, c2 = st.columns([1, 3])
            with c1:
                cls = "status-online" if status == "healthy" else "status-offline"
                st.markdown(f'<div class="status-indicator {cls}"></div>', unsafe_allow_html=True)
            with c2:
                st.write(f"API: {status.upper()}")
            if model_loaded:
                st.success("ML Model Loaded")
            else:
                st.warning("Rule-based Mode")
        else:
            st.error("API Unavailable")

        st.markdown("---")
        st.markdown("### Quick Actions")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Refresh", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        with c2:
            if st.button("Sample", use_container_width=True):
                st.session_state.run_sample = True

        st.markdown("---")
        st.info(
            f"Ocean Pollution Dashboard\n\n"
            f"Version: {APP_VERSION}\n\n"
            f"Updated: {datetime.now():%Y-%m-%d}"
        )


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
def render_dashboard() -> None:
    section_title("Ocean Pollution Monitoring Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Water Quality", "Real-time")
    with c2:
        metric_card("Predictions", str(len(st.session_state.predictions_history)))
    with c3:
        health = APIClient.get_health()
        api_status = "Online" if health and health.get("status") == "healthy" else "Offline"
        metric_card("API Status", api_status)
    with c4:
        metric_card("Last Update", datetime.now().strftime("%H:%M"))

    st.markdown("---")
    section_title("Quick Prediction", level=2)

    c1, c2, c3 = st.columns([2, 1, 2])
    with c1:
        quick_chl = st.slider(
            "Chlorophyll (CHL) mg/m3",
            0.0, 20.0, 2.5, 0.1, key="quick_chl",
        )
    with c2:
        st.write("")
        quick_predict = st.button("Predict Now", type="primary", use_container_width=True)
    with c3:
        if quick_predict:
            with st.spinner("Making prediction..."):
                result = APIClient.predict_single({"CHL": quick_chl})
            if result and result.get("success"):
                pred = result["prediction"]
                prediction_card(pred["level"], pred["confidence"])
                st.session_state.predictions_history.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "input": {"CHL": quick_chl},
                        "result": pred,
                    }
                )
            else:
                st.error("Prediction failed")

    st.markdown("---")

    if st.session_state.predictions_history:
        section_title("Recent Predictions", level=2)
        recent = st.session_state.predictions_history[-5:]
        for pred in reversed(recent):
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                t = datetime.fromisoformat(pred["timestamp"]).strftime("%H:%M")
                st.write(f"Time: {t}  |  CHL: {pred['input'].get('CHL', 'N/A')}")
            with c2:
                st.write(f"Level: {pred['result']['level']}")
            with c3:
                st.write(f"Confidence: {pred['result']['confidence']:.1%}")
            st.markdown("---")


# ---------------------------------------------------------------------------
# Real-time prediction
# ---------------------------------------------------------------------------
def _render_realtime_manual() -> None:
    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown("#### Essential Parameters")
        chl = st.number_input(
            "Chlorophyll (CHL) mg/m3 (required)",
            min_value=0.0, max_value=50.0, value=2.5, step=0.1,
            help="Primary indicator of algal biomass",
            key="chl_input",
        )
        st.markdown("#### Biological Parameters")
        diato = st.number_input(
            "Diatom Concentration (DIATO)", 0.0, 5.0, 0.05, 0.01,
            help="Diatom phytoplankton concentration",
        )
        dino = st.number_input(
            "Dinoflagellate Concentration (DINO)", 0.0, 5.0, 0.03, 0.01,
            help="Dinoflagellate phytoplankton concentration",
        )

    with c_right:
        st.markdown("#### Physical Parameters")
        pp = st.number_input(
            "Primary Production (PP)", 0.0, 2000.0, 350.0, 10.0,
            help="Primary production rate",
        )
        kd490 = st.number_input(
            "Diffuse Attenuation (KD490)", 0.0, 5.0, 0.25, 0.01,
            help="Diffuse attenuation coefficient at 490nm",
        )
        st.markdown("#### Chemical Parameters")
        cdm = st.number_input(
            "Colored Dissolved Matter (CDM)", 0.0, 1.0, 0.01, 0.001,
            help="Colored dissolved organic matter",
        )
        bbp = st.number_input(
            "Particulate Backscattering (BBP)", 0.0, 0.1, 0.001, 0.0001,
            help="Particulate backscattering coefficient",
        )

    st.markdown("---")
    st.markdown("### Prediction Settings")
    c1, c2 = st.columns(2)
    with c1:
        save_prediction = st.checkbox("Save to history", value=True)
        show_details = st.checkbox("Show detailed results", value=True)
    with c2:
        st.radio(
            "Mode:", ["Single Prediction", "Test Multiple Values"], horizontal=True
        )

    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        predict_button = st.button(
            "Run Prediction Analysis", type="primary", use_container_width=True
        )

    if not predict_button:
        return

    input_data = {
        "CHL": chl,
        "PP": pp if pp > 0 else None,
        "KD490": kd490 if kd490 > 0 else None,
        "DIATO": diato if diato > 0 else None,
        "DINO": dino if dino > 0 else None,
        "CDM": cdm if cdm > 0 else None,
        "BBP": bbp if bbp > 0 else None,
    }
    input_data = {k: v for k, v in input_data.items() if v is not None}

    with st.expander("Input Data Summary", expanded=True):
        st.dataframe(pd.DataFrame([input_data]), use_container_width=True)

    with st.spinner("Analyzing ocean parameters..."):
        result = APIClient.predict_single(input_data)

    if not (result and result.get("success")):
        st.error("Prediction failed. Please check your input and API connection.")
        if result and "error" in result:
            st.error(f"Error: {result['error']}")
        return

    prediction = result["prediction"]
    level = prediction["level"]
    confidence = prediction["confidence"]

    st.markdown("---")
    section_title("Prediction Results", level=2)

    c1, c2 = st.columns([1, 2])
    with c1:
        prediction_card(level, confidence)
        st.metric("Chlorophyll Value", f"{chl} mg/m3")
        st.metric("Features Used", prediction["features_used"])
        st.metric("Model Type", result.get("model_info", {}).get("model_type", "N/A"))

    with c2:
        st.markdown("#### Probability Distribution")
        probability_charts(prediction["probabilities"], prefix="rt")

    st.markdown("---")
    st.markdown("### Recommendations and Actions")
    recommendations(level)

    if save_prediction:
        st.session_state.predictions_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "input": input_data,
                "result": prediction,
                "model_info": result.get("model_info", {}),
            }
        )
        if len(st.session_state.predictions_history) > 100:
            st.session_state.predictions_history = st.session_state.predictions_history[-100:]
        st.toast("Prediction saved to history")

    if show_details:
        with st.expander("Detailed Results", expanded=False):
            st.json(result)


def _render_realtime_file_upload() -> None:
    uploaded_file = st.file_uploader(
        "Choose a CSV file with ocean data",
        type=["csv", "xlsx", "txt"],
        help="File should contain at least a 'CHL' column",
    )
    if not uploaded_file:
        return

    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file, sep="\t")
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return

    st.success(f"File loaded: {uploaded_file.name}  |  Rows: {len(df)}")
    with st.expander("Data Preview", expanded=True):
        st.dataframe(df.head(), use_container_width=True)

    if "CHL" not in df.columns:
        st.error("CSV must contain 'CHL' column")
        return

    if not st.button("Analyze All Samples", type="primary"):
        return

    with st.spinner("Processing data..."):
        samples = [
            {
                col: (float(row[col]) if pd.notnull(row[col]) else None)
                for col in ["CHL", "PP", "KD490", "DIATO", "DINO", "CDM", "BBP"]
                if col in row
            }
            for _, row in df.iterrows()
        ]
        result = APIClient.predict_batch({"samples": samples[:50]})

    if not (result and result.get("success")):
        st.error("Batch prediction failed")
        return

    st.success(f"Processed {len(result['results'])} samples")
    rows = []
    for i, res in enumerate(result["results"]):
        if res.get("success"):
            pred = res["prediction"]
            rows.append(
                {
                    "Sample": i + 1,
                    "CHL": samples[i].get("CHL", "N/A"),
                    "Level": pred["level"],
                    "Confidence": pred["confidence"],
                    "Features": pred["features_used"],
                }
            )

    if not rows:
        return

    results_df = pd.DataFrame(rows)
    st.dataframe(results_df, use_container_width=True)

    st.markdown("#### Analysis Statistics")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Samples", len(rows))
    with c2:
        st.metric("High Risk", len([r for r in rows if r["Level"] == "HIGH"]))
    with c3:
        st.metric("Avg Confidence", f"{results_df['Confidence'].mean():.1%}")

    st.markdown("#### Distribution Visualization")
    c1, c2 = st.columns(2)
    with c1:
        counts = results_df["Level"].value_counts()
        fig = px.pie(
            values=counts.values,
            names=counts.index,
            title="Pollution Level Distribution",
            color=counts.index,
            color_discrete_map=API.LEVEL_COLORS,
        )
        st.plotly_chart(fig, use_container_width=True, key="file_levels_chart")
    with c2:
        fig = px.histogram(
            results_df, x="Confidence", nbins=20,
            title="Confidence Distribution", color_discrete_sequence=["#1E88E5"],
        )
        st.plotly_chart(fig, use_container_width=True, key="file_conf_chart")

    st.download_button(
        "Download Results as CSV",
        data=results_df.to_csv(index=False),
        file_name="pollution_analysis_results.csv",
        mime="text/csv",
    )


def _render_realtime_sample() -> None:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Example Data Sets")
        example_type = st.selectbox(
            "Select Example Scenario:",
            ["Clear Ocean", "Coastal Waters", "Polluted Bay", "Algal Bloom", "Custom"],
        )
        examples = {
            "Clear Ocean": {"CHL": 0.3, "PP": 150, "KD490": 0.1, "DIATO": 0.01},
            "Coastal Waters": {"CHL": 2.5, "PP": 350, "KD490": 0.25, "DIATO": 0.05},
            "Polluted Bay": {"CHL": 8.0, "PP": 800, "KD490": 0.5, "DIATO": 0.2},
            "Algal Bloom": {"CHL": 15.0, "PP": 1200, "KD490": 0.8, "DIATO": 0.5},
            "Custom": {},
        }
        if example_type != "Custom":
            example_data = examples[example_type]
            for k, v in example_data.items():
                st.write(f"- {k}: {v}")
            if st.button(f"Test {example_type} Scenario"):
                st.session_state.sample_data = example_data
                st.rerun()

    with c2:
        st.markdown("#### Quick Test")
        if st.button("Get Sample Prediction from API"):
            with st.spinner("Getting sample prediction..."):
                result = APIClient.get_sample_prediction()
            if result and result.get("success"):
                st.success("Sample prediction received")
                sample = result.get("sample_data", {})
                pred = result.get("result", {}).get("prediction", {})
                c1_, c2_ = st.columns(2)
                with c1_:
                    st.markdown("Sample Data:")
                    for k, v in sample.items():
                        st.write(f"- {k}: {v}")
                with c2_:
                    if pred:
                        st.markdown("Prediction:")
                        st.write(f"- Level: {pred.get('level', 'N/A')}")
                        st.write(f"- Confidence: {pred.get('confidence', 0):.1%}")
                        st.write(f"- CHL: {pred.get('chl_value', 'N/A')}")

    st.markdown("---")
    st.markdown("#### Test Multiple Values")
    test_values = st.text_area(
        "Enter multiple CHL values (comma separated):",
        "0.5, 1.0, 2.5, 5.0, 10.0",
    )
    if st.button("Run Multiple Tests"):
        try:
            values = [float(x.strip()) for x in test_values.split(",") if x.strip()]
            progress = st.progress(0)
            rows = []
            for i, v in enumerate(values):
                result = APIClient.predict_single({"CHL": v})
                if result and result.get("success"):
                    pred = result["prediction"]
                    rows.append(
                        {
                            "CHL": v,
                            "Level": pred["level"],
                            "Confidence": pred["confidence"],
                            "Color": API.LEVEL_COLORS.get(pred["level"], "#999"),
                        }
                    )
                progress.progress((i + 1) / len(values))
            progress.empty()

            if rows:
                df_test = pd.DataFrame(rows)
                fig = px.scatter(
                    df_test, x="CHL", y="Confidence", color="Level",
                    title="CHL vs Confidence",
                    color_discrete_map=API.LEVEL_COLORS,
                    size=[20] * len(df_test),
                    hover_data=["Level", "Confidence"],
                )
                fig.update_layout(
                    xaxis_title="Chlorophyll (CHL) mg/m3",
                    yaxis_title="Confidence", height=400,
                )
                st.plotly_chart(fig, use_container_width=True, key="test_scatter_chart")
                st.dataframe(
                    df_test[["CHL", "Level", "Confidence"]], use_container_width=True
                )
        except ValueError as e:
            st.error(f"Invalid input: {e}")


def render_realtime_prediction() -> None:
    section_title("Real-time Pollution Prediction")
    tab1, tab2, tab3 = st.tabs(["Manual Input", "File Upload", "Sample Data"])
    with tab1:
        _render_realtime_manual()
    with tab2:
        _render_realtime_file_upload()
    with tab3:
        _render_realtime_sample()


# ---------------------------------------------------------------------------
# Batch analysis
# ---------------------------------------------------------------------------
def render_batch_analysis() -> None:
    section_title("Batch Data Analysis")
    tab1, tab2, tab3 = st.tabs(
        ["Time Series", "Spatial Analysis", "Statistical Analysis"]
    )

    with tab1:
        st.markdown("### Time Series Analysis")
        rng = np.random.default_rng(42)
        dates = pd.date_range("2024-01-01", periods=365, freq="D")
        chl = rng.uniform(0.1, 12, len(dates)) + np.sin(np.linspace(0, 4 * np.pi, len(dates))) * 2
        ts_df = pd.DataFrame(
            {
                "Date": dates,
                "CHL": chl,
                "PP": chl * 100 + rng.normal(0, 20, len(dates)),
                "KD490": 0.1 + chl * 0.05 + rng.normal(0, 0.02, len(dates)),
            }
        )
        fig = px.line(ts_df, x="Date", y="CHL", title="Chlorophyll Over Time")
        st.plotly_chart(fig, use_container_width=True, key="ts_line_chart")

        if st.button("Analyze Time Series", type="primary"):
            with st.spinner("Analyzing..."):
                sample_dates, sample_chl = dates[::30], chl[::30]
                preds = [
                    "LOW" if c < 1 else "MEDIUM" if c < 5 else "HIGH" for c in sample_chl
                ]
                pred_df = pd.DataFrame(
                    {"Date": sample_dates, "CHL": sample_chl, "Prediction": preds}
                )
                fig = px.scatter(
                    pred_df, x="Date", y="CHL", color="Prediction",
                    title="Pollution Predictions Over Time",
                    color_discrete_map=API.LEVEL_COLORS,
                    size=[10] * len(pred_df),
                    hover_data=["Prediction"],
                )
                st.plotly_chart(fig, use_container_width=True, key="ts_pred_chart")

    with tab2:
        st.markdown("### Spatial Analysis")
        rng = np.random.default_rng(42)
        n = 50
        spatial = pd.DataFrame(
            {
                "Latitude": rng.uniform(25, 45, n),
                "Longitude": rng.uniform(-80, -60, n),
                "CHL": rng.uniform(0.1, 10, n),
            }
        )
        spatial["Pollution_Level"] = pd.cut(
            spatial["CHL"], bins=[0, 1, 3, 100], labels=["LOW", "MEDIUM", "HIGH"]
        )
        fig = px.scatter_mapbox(
            spatial,
            lat="Latitude",
            lon="Longitude",
            color="Pollution_Level",
            size="CHL",
            hover_data=["CHL", "Pollution_Level"],
            color_discrete_map=API.LEVEL_COLORS,
            zoom=3,
            height=500,
            title="Pollution Hotspots Map",
        )
        fig.update_layout(mapbox_style="open-street-map", margin=dict(r=0, t=50, l=0, b=0))
        st.plotly_chart(fig, use_container_width=True, key="spatial_map")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Points", n)
        with c2:
            st.metric("High Risk Areas", len(spatial[spatial["Pollution_Level"] == "HIGH"]))
        with c3:
            st.metric("Average CHL", f"{spatial['CHL'].mean():.2f}")

    with tab3:
        st.markdown("### Statistical Analysis")
        history = st.session_state.predictions_history
        if not history:
            st.info("Make some predictions first to see statistical analysis")
            return

        rows = [
            {
                "CHL": e["input"].get("CHL", 0),
                "Level": e["result"].get("level", "UNKNOWN"),
                "Confidence": e["result"].get("confidence", 0),
            }
            for e in history
            if e.get("result")
        ]
        if not rows:
            st.info("No valid predictions in history")
            return

        stats_df = pd.DataFrame(rows)
        st.markdown("#### Descriptive Statistics")
        st.dataframe(stats_df.describe(), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(
                stats_df, x="CHL", nbins=20,
                title="CHL Value Distribution",
                color_discrete_sequence=["#1E88E5"],
            )
            st.plotly_chart(fig, use_container_width=True, key="stats_chl_dist")
        with c2:
            counts = stats_df["Level"].value_counts()
            fig = px.bar(
                x=counts.index, y=counts.values,
                title="Prediction Level Distribution",
                color=counts.index,
                color_discrete_map=API.LEVEL_COLORS,
            )
            st.plotly_chart(fig, use_container_width=True, key="stats_level_dist")


# ---------------------------------------------------------------------------
# Data insights
# ---------------------------------------------------------------------------
def render_data_insights() -> None:
    section_title("Data Insights and Visualizations")
    tab1, tab2, tab3 = st.tabs(
        ["Pollution Trends", "Environmental Factors", "Prediction Patterns"]
    )

    with tab1:
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly_avg = {
            "LOW": [45, 42, 38, 35, 30, 25, 20, 22, 28, 35, 40, 43],
            "MEDIUM": [35, 38, 40, 42, 45, 48, 50, 48, 45, 42, 38, 36],
            "HIGH": [20, 20, 22, 23, 25, 27, 30, 30, 27, 23, 22, 21],
        }

        fig = go.Figure()
        for level in API.POLLUTION_LEVELS:
            fig.add_trace(
                go.Scatter(
                    x=months,
                    y=monthly_avg[level],
                    name=level,
                    mode="lines+markers",
                    line=dict(color=API.LEVEL_COLORS[level], width=3),
                    marker=dict(size=8),
                )
            )
        fig.update_layout(
            title="Monthly Pollution Level Trends",
            xaxis_title="Month",
            yaxis_title="Percentage (%)",
            height=500,
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True, key="insights_trend_chart")

        st.markdown("#### Insights")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                "Seasonal Patterns:\n"
                "- HIGH pollution peaks in summer months\n"
                "- LOW pollution highest in winter\n"
                "- MEDIUM levels relatively stable"
            )
        with c2:
            st.markdown(
                "Recommendations:\n"
                "- Increase monitoring in summer\n"
                "- Plan cleanup operations in advance\n"
                "- Adjust thresholds seasonally"
            )

    with tab2:
        st.markdown("### Environmental Factor Correlations")
        factors = ["Temperature", "Salinity", "pH", "Oxygen", "Nutrients", "Turbidity"]
        correlations = {
            "CHL": [0.65, -0.32, 0.28, -0.45, 0.72, 0.58],
            "PP": [0.58, -0.25, 0.31, -0.38, 0.68, 0.52],
            "KD490": [0.42, -0.18, 0.15, -0.28, 0.45, 0.78],
        }
        fig = px.imshow(
            pd.DataFrame(correlations, index=factors),
            text_auto=".2f",
            title="Environmental Factor Correlations",
            color_continuous_scale="RdBu",
            aspect="auto",
        )
        st.plotly_chart(fig, use_container_width=True, key="insights_heatmap")

        st.markdown("#### Factor Importance")
        importance_data = pd.DataFrame(
            {
                "Factor": factors,
                "Importance": [0.25, 0.15, 0.10, 0.20, 0.18, 0.12],
            }
        ).sort_values("Importance", ascending=True)
        fig = px.bar(
            importance_data,
            x="Importance",
            y="Factor",
            orientation="h",
            title="Environmental Factor Importance",
            color="Importance",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig, use_container_width=True, key="insights_importance")

    with tab3:
        st.markdown("### Prediction Pattern Analysis")
        patterns = {
            "Clear Pattern": {"LOW": 70, "MEDIUM": 20, "HIGH": 10},
            "Moderate Pattern": {"LOW": 30, "MEDIUM": 50, "HIGH": 20},
            "Polluted Pattern": {"LOW": 10, "MEDIUM": 30, "HIGH": 60},
            "Variable Pattern": {"LOW": 40, "MEDIUM": 40, "HIGH": 20},
        }
        fig = go.Figure()
        for name, data in patterns.items():
            fig.add_trace(
                go.Scatterpolar(
                    r=[data["LOW"], data["MEDIUM"], data["HIGH"]],
                    theta=["LOW", "MEDIUM", "HIGH"],
                    name=name,
                    fill="toself",
                )
            )
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="Prediction Pattern Analysis",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True, key="insights_radar")


# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------
def render_model_info() -> None:
    section_title("Model Information and Diagnostics")

    model_info = APIClient.get_model_info()
    if not (model_info and model_info.get("success")):
        st.error("Could not retrieve model information")
        st.info("Make sure the API is running and accessible")
        return

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Model Status")
        model_data = model_info.get("model", {})
        metadata = model_info.get("metadata", {})

        if model_data.get("loaded"):
            st.success("ML Model Loaded")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Features", model_data.get("features_count", 0))
            with m2:
                accuracy = metadata.get("accuracy", 0)
                st.metric("Accuracy", f"{float(accuracy):.1%}" if accuracy else "N/A")
            with m3:
                st.metric("Training Samples", f"{metadata.get('samples', 0):,}")

            with st.expander("Model Details", expanded=True):
                st.json(metadata, expanded=False)

            with st.expander("Feature List", expanded=False):
                features = model_data.get("features", [])
                for i, feat in enumerate(features[:20], 1):
                    st.write(f"{i}. {feat}")
                if len(features) > 20:
                    st.write(f"... and {len(features) - 20} more features")
        else:
            st.warning("Rule-based Model Active")
            st.info("Using simple threshold-based prediction")

    with c2:
        st.markdown("### Model Diagnostics")
        st.markdown("#### Model Testing")
        test_chl = st.number_input("Test CHL Value", value=2.5, key="test_chl")

        if st.button("Test Model"):
            with st.spinner("Testing model..."):
                result = APIClient.predict_single({"CHL": test_chl})
            if result and result.get("success"):
                prediction = result["prediction"]
                c1_, c2_ = st.columns(2)
                with c1_:
                    st.success(prediction["level"])
                    st.write(f"Confidence: {prediction['confidence']:.1%}")
                with c2_:
                    st.write(f"Model: {result.get('model_info', {}).get('model_type', 'N/A')}")
                    st.write(f"Features: {prediction['features_used']}")
            else:
                st.error("Test failed")

        st.markdown("---")
        st.markdown("#### System Information")
        sys_info = {
            "Python Version": sys.version.split()[0],
            "Platform": sys.platform,
            "API Base URL": API.BASE_URL,
            "Current Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        for k, v in sys_info.items():
            st.write(f"**{k}:** {v}")


# ---------------------------------------------------------------------------
# Prediction history
# ---------------------------------------------------------------------------
def render_prediction_history() -> None:
    section_title("Prediction History")

    history = st.session_state.predictions_history
    if not history:
        st.info("No prediction history available. Make some predictions first.")
        return

    total = len(history)
    successful = len([h for h in history if h.get("result")])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Predictions", total)
    with c2:
        st.metric("Successful", successful)
    with c3:
        st.metric("Success Rate", f"{successful / total:.1%}" if total else "0%")
    with c4:
        high = len([h for h in history if h.get("result", {}).get("level") == "HIGH"])
        st.metric("High Risk", high)

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        filter_level = st.multiselect(
            "Filter by Level:", options=list(API.POLLUTION_LEVELS), default=[]
        )
    with c2:
        min_confidence = st.slider("Minimum Confidence:", 0.0, 1.0, 0.0, 0.1)
    with c3:
        items_per_page = st.selectbox("Items per page:", [10, 25, 50, 100], index=0)

    filtered = list(history)
    if filter_level:
        filtered = [h for h in filtered if h.get("result", {}).get("level") in filter_level]
    if min_confidence > 0:
        filtered = [
            h for h in filtered
            if h.get("result", {}).get("confidence", 0) >= min_confidence
        ]

    total_filtered = len(filtered)
    max_page = max(1, (total_filtered + items_per_page - 1) // items_per_page)
    page_number = st.number_input("Page:", 1, max_page, 1, 1)

    start_idx = (page_number - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, total_filtered)

    st.markdown(f"Showing {start_idx + 1}-{end_idx} of {total_filtered} predictions")

    for i in range(start_idx, end_idx):
        entry = filtered[i]
        result = entry.get("result", {})
        with st.expander(
            f"Prediction {i + 1}: {result.get('level', 'UNKNOWN')} - "
            f"{result.get('confidence', 0):.1%}",
            expanded=False,
        ):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.write("**Time:**")
                st.write(datetime.fromisoformat(entry["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"))
                st.write("**Input Values:**")
                for k, v in entry["input"].items():
                    st.write(f"- {k}: {v}")
            with c2:
                st.write("**Prediction:**")
                st.write(f"Level: **{result.get('level', 'UNKNOWN')}**")
                st.write(f"Confidence: **{result.get('confidence', 0):.1%}**")
                st.write(f"CHL Value: **{result.get('chl_value', 'N/A')}**")
                st.write("**Probabilities:**")
                for lvl, prob in result.get("probabilities", {}).items():
                    st.write(f"- {lvl}: {prob:.1%}")
            with c3:
                st.write("**Model Info:**")
                st.write(f"Features Used: {result.get('features_used', 'N/A')}")
                st.write(f"Model Type: {entry.get('model_info', {}).get('model_type', 'N/A')}")
                a1, a2 = st.columns(2)
                with a1:
                    if st.button("Re-run", key=f"rerun_{i}"):
                        st.session_state.rerun_data = entry["input"]
                        st.rerun()
                with a2:
                    if st.button("Delete", key=f"delete_{i}"):
                        idx = st.session_state.predictions_history.index(entry)
                        st.session_state.predictions_history.pop(idx)
                        st.rerun()

    st.markdown("---")
    st.markdown("### Export Options")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Export to CSV", use_container_width=True):
            export = [
                {
                    "timestamp": e["timestamp"],
                    "CHL": e["input"].get("CHL", ""),
                    "PP": e["input"].get("PP", ""),
                    "KD490": e["input"].get("KD490", ""),
                    "level": e["result"].get("level", ""),
                    "confidence": e["result"].get("confidence", ""),
                    "features_used": e["result"].get("features_used", ""),
                }
                for e in filtered
            ]
            if export:
                st.download_button(
                    "Download CSV",
                    data=pd.DataFrame(export).to_csv(index=False),
                    file_name="prediction_history.csv",
                    mime="text/csv",
                    key="download_csv",
                )
    with c2:
        if st.button("Clear History", use_container_width=True):
            st.session_state.predictions_history = []
            st.rerun()
    with c3:
        if st.button("Show Charts", use_container_width=True) and filtered:
            rows = [
                {
                    "time": datetime.fromisoformat(e["timestamp"]),
                    "level": e["result"].get("level", "UNKNOWN"),
                    "confidence": e["result"].get("confidence", 0),
                    "CHL": e["input"].get("CHL", 0),
                }
                for e in filtered
            ]
            df_hist = pd.DataFrame(rows)
            fig = px.scatter(
                df_hist, x="time", y="CHL", color="level",
                title="Prediction History Timeline",
                color_discrete_map=API.LEVEL_COLORS,
                hover_data=["confidence", "level"],
            )
            st.plotly_chart(fig, use_container_width=True, key="history_timeline")


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
def render_settings() -> None:
    section_title("Settings and Configuration")
    tab1, tab2, tab3 = st.tabs(["API Settings", "Display Settings", "Advanced"])

    with tab1:
        st.markdown("### API Configuration")
        current_url = st.text_input(
            "API Base URL:",
            value=API.BASE_URL,
            help="URL of the Ocean Pollution API server",
        )
        c1, c2 = st.columns([3, 1])
        with c1:
            if st.button("Test API Connection", use_container_width=True):
                try:
                    r = requests.get(f"{current_url}/health", timeout=5)
                    if r.status_code == 200:
                        st.success("API connection successful")
                        st.json(r.json(), expanded=False)
                    else:
                        st.error(f"API error: {r.status_code}")
                except Exception as e:
                    st.error(f"Connection failed: {e}")
        with c2:
            if st.button("Reset to Default", use_container_width=True):
                st.rerun()

        st.markdown("### Available Endpoints")
        for endpoint, description in [
            ("/", "API Information"),
            ("/docs", "Interactive Documentation"),
            ("/health", "Health Check"),
            ("/model-info", "Model Information"),
            ("/predict", "Single Prediction"),
            ("/predict/batch", "Batch Prediction"),
            ("/project/structure", "Project Structure"),
        ]:
            st.write(f"`{endpoint}` - {description}")

    with tab2:
        st.markdown("### Display Settings")
        st.selectbox("Color Theme:", ["Default", "Dark", "Light", "Ocean", "Forest"], index=0)
        st.selectbox("Chart Style:", ["Plotly", "Matplotlib", "Altair"], index=0)

        c1, c2 = st.columns(2)
        with c1:
            auto_refresh = st.checkbox("Auto-refresh data", value=False)
            if auto_refresh:
                interval = st.slider("Refresh interval (seconds):", 5, 300, 60, 5)
                st.info(f"Data will refresh every {interval} seconds")
        with c2:
            st.checkbox("Show animations", value=True)
            st.checkbox("Compact mode", value=False)

        if st.button("Apply Display Settings", type="primary"):
            st.success("Display settings applied")

    with tab3:
        st.markdown("### Advanced Settings")
        st.checkbox("Enable debug mode", value=False)
        st.selectbox("Log Level:", ["INFO", "DEBUG", "WARNING", "ERROR"], index=0)

        st.markdown("#### Cache Settings")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Clear Cache", use_container_width=True):
                st.cache_data.clear()
                st.success("Cache cleared")
        with c2:
            if st.button("Clear Session State", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.success("Session state cleared")
                st.rerun()

        st.markdown("#### System Information")
        for k, v in {
            "Streamlit Version": st.__version__,
            "Python Version": platform.python_version(),
            "Operating System": platform.system(),
            "Processor": platform.processor(),
            "Working Directory": os.getcwd(),
        }.items():
            st.text(f"{k}: {v}")


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 600;
        border-bottom: 3px solid #1E88E5;
        padding-bottom: 0.5rem;
    }
    .prediction-card {
        padding: 1.8rem;
        border-radius: 15px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        border-left: 6px solid;
        transition: transform 0.3s ease;
    }
    .prediction-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.15);
    }
    .low-pollution    { background: linear-gradient(135deg,#E8F5E9,#C8E6C9); border-left-color:#4CAF50; }
    .medium-pollution { background: linear-gradient(135deg,#FFF3E0,#FFE0B2); border-left-color:#FF9800; }
    .high-pollution   { background: linear-gradient(135deg,#FFEBEE,#FFCDD2); border-left-color:#F44336; }
    .metric-card {
        padding: 1.2rem;
        border-radius: 10px;
        background: linear-gradient(135deg,#f8f9fa,#e9ecef);
        text-align: center;
        border: 2px solid #dee2e6;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: #1E88E5;
        background: linear-gradient(135deg,#e3f2fd,#bbdefb);
    }
    .status-indicator {
        display: inline-block;
        width: 14px;
        height: 14px;
        border-radius: 50%;
        margin-right: 10px;
        animation: pulse 2s infinite;
    }
    .status-online  { background-color:#4CAF50; box-shadow:0 0 0 0 rgba(76,175,80,.7); }
    .status-offline { background-color:#F44336; box-shadow:0 0 0 0 rgba(244,67,54,.7); }
    @keyframes pulse {
        0%   { box-shadow: 0 0 0 0 rgba(76,175,80,.7); }
        70%  { box-shadow: 0 0 0 10px rgba(76,175,80,0); }
        100% { box-shadow: 0 0 0 0 rgba(76,175,80,0); }
    }
    .stButton > button { border-radius: 8px; font-weight: 600; transition: all .3s ease; }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,.2); }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: 10px 20px; font-weight: 600; }
    .dataframe { border-radius: 8px; overflow: hidden; }
</style>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Ocean Pollution Dashboard",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    init_session_state()
    render_sidebar()

    tab = st.session_state.current_tab
    if tab == "Dashboard":
        render_dashboard()
    elif tab == "Real-time Prediction":
        render_realtime_prediction()
    elif tab == "Batch Analysis":
        render_batch_analysis()
    elif tab == "Data Insights":
        render_data_insights()
    elif tab == "Model Information":
        render_model_info()
    elif tab == "Prediction History":
        render_prediction_history()
    elif tab == "Settings":
        render_settings()

    st.markdown("---")
    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.markdown(
            '<div style="text-align:center;color:#666;font-size:0.9rem;">'
            f"Ocean Pollution Dashboard v{APP_VERSION} | "
            "Built for Environmental Protection"
            "</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
