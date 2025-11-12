# src/train_classification.py
import pandas as pd, numpy as np, os, joblib
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import shap, matplotlib.pyplot as plt

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# load labeled dataset (if you haven't labeled, run label script or use weak labels)
df = pd.read_csv('data/processed/dataset_labeled.csv', parse_dates=['time_utc']) if os.path.exists('data/processed/dataset_labeled.csv') else pd.read_csv('data/processed/dataset_ready.csv', parse_dates=['time_utc'])
# If no label column, create weak label: (simple rule) (this is fallback)
if 'label_polluted' not in df.columns:
    df['label_polluted'] = ((df.get('sss_grad_per_km',0) > df.get('sss_grad_per_km',0).quantile(0.9)) & (df.get('sst',0) > df.get('sst',0).median())).astype(int)

feature_cols = ['sst','sss','sss_grad_per_km','dayofyear','hour']
for c in feature_cols:
    if c not in df.columns:
        df[c] = 0
X = df[feature_cols].fillna(df[feature_cols].median())
y = df['label_polluted'].astype(int)

# spatial groups via kmeans on coords
coords = df[['lat','lon']].fillna(0)
k = min(5, len(df))
kmeans = KMeans(n_clusters=k, random_state=42).fit(coords)
groups = kmeans.labels_

gkf = GroupKFold(n_splits=min(5,k))
probs = np.zeros(len(df))
aucs = []
for train_idx, test_idx in gkf.split(X, y, groups=groups):
    Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
    ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
    model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss')
    model.fit(Xtr, ytr)
    prob = model.predict_proba(Xte)[:,1]
    probs[test_idx] = prob
    auc = roc_auc_score(yte, prob) if len(np.unique(yte))>1 else np.nan
    aucs.append(auc)
    print("Fold AUC:", auc)

print("Mean AUC:", np.nanmean(aucs))

# final train
final = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.03, use_label_encoder=False, eval_metric='logloss')
final.fit(X, y)
joblib.dump(final, 'models/xgb_pollution_classifier.joblib')

# SHAP
explainer = shap.TreeExplainer(final)
shap_values = explainer.shap_values(X)
plt.figure(figsize=(7,5))
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig('results/shap_summary_classification.png', dpi=200)

df['pred_prob'] = probs
df['pred_label'] = (probs>0.5).astype(int)
df.to_csv('results/predictions_with_probs.csv', index=False)
print("Saved model and results.")
