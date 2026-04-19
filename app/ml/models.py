"""
ML Model Registry
-----------------
Five pre-built, self-contained model runners for the Amazon Products dataset.
Each function accepts a DataFrame, runs end-to-end, saves artifacts to
WORKING_DIR, and returns a structured metrics dict.

The ModelSelector agent picks which function to call based on the task prompt.
The MLAnalyst agent executes the chosen function via the code execution tool.
"""

import os
import json
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    accuracy_score, classification_report,
    silhouette_score
)

warnings.filterwarnings("ignore")

# Resolved at import time so every function can use it
from app.core.config import WORKING_DIR


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode all object/category columns in-place copy."""
    df = df.copy()
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df


def _drop_useless(df: pd.DataFrame) -> pd.DataFrame:
    """Drop constant columns and columns that are >70% null."""
    df = df.copy()
    df = df.loc[:, df.isnull().mean() < 0.70]
    df = df.loc[:, df.nunique() > 1]
    return df


def _save_fig(filename: str) -> str:
    path = os.path.join(WORKING_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
    return path


# ---------------------------------------------------------------------------
# 1. Linear Regression
#    Best for: "predict a number simply / interpretably"
#    e.g. "predict discounted_price from rating and category"
# ---------------------------------------------------------------------------

def run_linear_regression(df: pd.DataFrame, target_col: str) -> dict:
    """
    Fits a Linear Regression to predict `target_col`.
    Saves a predicted-vs-actual scatter plot.
    Returns MAE, R², and top feature coefficients.
    """
    df = _drop_useless(df)
    df = df.dropna(subset=[target_col])
    df = _encode_categoricals(df)
    df = df.select_dtypes(include="number").dropna()

    if target_col not in df.columns:
        return {"error": f"target column '{target_col}' not found after preprocessing"}
    if df.shape[1] < 2:
        return {"error": "not enough numeric features after preprocessing"}

    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Plot: predicted vs actual
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, preds, alpha=0.3, edgecolors="none", color="#378ADD")
    lim = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    plt.plot(lim, lim, "r--", linewidth=1)
    plt.xlabel(f"Actual {target_col}")
    plt.ylabel(f"Predicted {target_col}")
    plt.title(f"Linear Regression — predicted vs actual ({target_col})")
    artifact = _save_fig("linear_regression_pred_vs_actual.png")

    # Top 10 coefficients by magnitude
    coef_series = pd.Series(model.coef_, index=X.columns)
    top_coef = coef_series.abs().nlargest(10).index
    top_coef_vals = coef_series[top_coef].to_dict()

    metrics = {
        "model": "linear_regression",
        "target_col": target_col,
        "mae": round(float(mean_absolute_error(y_test, preds)), 4),
        "r2": round(float(r2_score(y_test, preds)), 4),
        "top_coefficients": {k: round(v, 4) for k, v in top_coef_vals.items()},
        "artifact": artifact,
    }
    print(json.dumps(metrics, indent=2))
    return metrics


# ---------------------------------------------------------------------------
# 2. Random Forest Classifier
#    Best for: "predict a binary flag / category"
#    e.g. "predict is_best_seller", "what makes a product get a coupon"
# ---------------------------------------------------------------------------

def run_random_forest_classifier(df: pd.DataFrame, target_col: str) -> dict:
    """
    Fits a Random Forest Classifier to predict `target_col`.
    Saves a feature importance bar chart.
    Returns accuracy, classification report, and top features.
    """
    df = _drop_useless(df)
    df = df.dropna(subset=[target_col])
    df = _encode_categoricals(df)
    df = df.select_dtypes(include="number").dropna()

    if target_col not in df.columns:
        return {"error": f"target column '{target_col}' not found after preprocessing"}

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Feature importance chart
    importances = pd.Series(model.feature_importances_, index=X.columns)
    top10 = importances.nlargest(10).sort_values()
    plt.figure(figsize=(7, 5))
    top10.plot(kind="barh", color="#1D9E75")
    plt.xlabel("Feature importance")
    plt.title(f"Random Forest — top features for predicting {target_col}")
    artifact = _save_fig("rf_classifier_feature_importance.png")

    report = classification_report(y_test, preds, output_dict=True)
    metrics = {
        "model": "random_forest_classifier",
        "target_col": target_col,
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "precision": round(float(report.get("weighted avg", {}).get("precision", 0)), 4),
        "recall": round(float(report.get("weighted avg", {}).get("recall", 0)), 4),
        "f1": round(float(report.get("weighted avg", {}).get("f1-score", 0)), 4),
        "top_features": {k: round(float(v), 4) for k, v in importances.nlargest(5).items()},
        "artifact": artifact,
    }
    print(json.dumps(metrics, indent=2))
    return metrics


# ---------------------------------------------------------------------------
# 3. KMeans Clustering
#    Best for: "group / segment / find patterns"
#    e.g. "segment products by price and rating"
# ---------------------------------------------------------------------------

def run_kmeans(df: pd.DataFrame, features: list, k: int = 4) -> dict:
    """
    Clusters products on `features` using KMeans(k).
    Saves a scatter plot (first two features) coloured by cluster.
    Returns inertia, silhouette score, and per-cluster summary.
    """
    df = _drop_useless(df)
    available = [f for f in features if f in df.columns]
    if len(available) < 2:
        return {"error": f"need at least 2 valid features, got: {available}"}

    data = df[available].dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    model = KMeans(n_clusters=k, random_state=42, n_init="auto")
    data = data.copy()
    data["cluster"] = model.fit_predict(scaled)

    sil = silhouette_score(scaled, data["cluster"]) if k > 1 else 0.0

    # Scatter on first two features
    plt.figure(figsize=(7, 5))
    for cluster_id in range(k):
        subset = data[data["cluster"] == cluster_id]
        plt.scatter(
            subset[available[0]], subset[available[1]],
            alpha=0.4, edgecolors="none", label=f"Cluster {cluster_id}"
        )
    plt.xlabel(available[0])
    plt.ylabel(available[1])
    plt.title(f"KMeans (k={k}) — {available[0]} vs {available[1]}")
    plt.legend(markerscale=1.5, fontsize=8)
    artifact = _save_fig("kmeans_clusters.png")

    # Per-cluster summary (mean of original features)
    cluster_summary = (
        data.groupby("cluster")[available]
        .mean()
        .round(3)
        .to_dict(orient="index")
    )

    metrics = {
        "model": "kmeans",
        "k": k,
        "features": available,
        "inertia": round(float(model.inertia_), 2),
        "silhouette_score": round(float(sil), 4),
        "cluster_summary": {str(k_): v for k_, v in cluster_summary.items()},
        "artifact": artifact,
    }
    print(json.dumps(metrics, indent=2))
    return metrics


# ---------------------------------------------------------------------------
# 4. Isolation Forest (anomaly detection)
#    Best for: "find outliers / unusual / suspicious products"
#    e.g. "find products with unusual price-to-rating ratios"
# ---------------------------------------------------------------------------

def run_isolation_forest(df: pd.DataFrame, features: list = None) -> dict:
    """
    Detects anomalous products using Isolation Forest.
    Saves a scatter plot highlighting anomalies vs normal products.
    Returns anomaly count, contamination rate, and sample anomalous rows.
    """
    df = _drop_useless(df)

    if features:
        available = [f for f in features if f in df.columns]
    else:
        # Default: all numeric columns
        available = df.select_dtypes(include="number").columns.tolist()

    if len(available) < 1:
        return {"error": "no valid numeric features found"}

    data = df[available].dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    model = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    data = data.copy()
    data["anomaly"] = model.fit_predict(scaled)  # -1 = anomaly, 1 = normal

    n_anomalies = int((data["anomaly"] == -1).sum())
    anomaly_rate = round(n_anomalies / len(data), 4)

    # Scatter on first two features, coloured by anomaly status
    f0, f1 = available[0], available[min(1, len(available) - 1)]
    normal = data[data["anomaly"] == 1]
    anomalous = data[data["anomaly"] == -1]

    plt.figure(figsize=(7, 5))
    plt.scatter(normal[f0], normal[f1], alpha=0.2, color="#378ADD",
                edgecolors="none", label="Normal")
    plt.scatter(anomalous[f0], anomalous[f1], alpha=0.7, color="#E24B4A",
                edgecolors="none", label=f"Anomaly ({n_anomalies})")
    plt.xlabel(f0)
    plt.ylabel(f1)
    plt.title(f"Isolation Forest — anomaly detection ({f0} vs {f1})")
    plt.legend(fontsize=8)
    artifact = _save_fig("isolation_forest_anomalies.png")

    # Sample of anomalous rows (original df, unscaled)
    anomaly_indices = data[data["anomaly"] == -1].index
    sample = df.loc[anomaly_indices].head(5).to_dict(orient="records")

    metrics = {
        "model": "isolation_forest",
        "features_used": available,
        "total_products": len(data),
        "anomalies_found": n_anomalies,
        "anomaly_rate": anomaly_rate,
        "sample_anomalies": sample,
        "artifact": artifact,
    }
    print(json.dumps(metrics, indent=2))
    return metrics


# ---------------------------------------------------------------------------
# 5. XGBoost + SHAP
#    Best for: "what drives X / explain why / feature importance"
#    e.g. "what factors drive purchased_last_month"
# ---------------------------------------------------------------------------

def run_xgboost_shap(df: pd.DataFrame, target_col: str) -> dict:
    """
    Fits an XGBoost regressor and computes SHAP values for explainability.
    Saves a SHAP summary bar chart.
    Returns MAE, R², and ranked SHAP feature importances.

    Falls back to RandomForestRegressor if xgboost is not installed.
    """
    df = _drop_useless(df)
    df = df.dropna(subset=[target_col])
    df = _encode_categoricals(df)
    df = df.select_dtypes(include="number").dropna()

    if target_col not in df.columns:
        return {"error": f"target column '{target_col}' not found after preprocessing"}

    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Try XGBoost, fall back to RandomForest
    try:
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=100, random_state=42,
                             verbosity=0, n_jobs=-1)
        model_name = "xgboost"
    except ImportError:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model_name = "random_forest_regressor (xgboost not installed)"

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Try SHAP values
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        mean_abs_shap = pd.Series(
            np.abs(shap_values).mean(axis=0), index=X.columns
        ).sort_values(ascending=False)

        # SHAP bar chart
        top = mean_abs_shap.head(10).sort_values()
        plt.figure(figsize=(7, 5))
        top.plot(kind="barh", color="#534AB7")
        plt.xlabel("Mean |SHAP value|")
        plt.title(f"SHAP feature importance — {target_col}")
        artifact = _save_fig("xgboost_shap_importance.png")
        shap_dict = {k: round(float(v), 4) for k, v in mean_abs_shap.head(10).items()}

    except ImportError:
        # SHAP not installed — use built-in feature importances
        if hasattr(model, "feature_importances_"):
            importances = pd.Series(model.feature_importances_, index=X.columns)
        else:
            importances = pd.Series(
                np.abs(model.coef_) if hasattr(model, "coef_") else np.ones(X.shape[1]),
                index=X.columns
            )
        top = importances.nlargest(10).sort_values()
        plt.figure(figsize=(7, 5))
        top.plot(kind="barh", color="#534AB7")
        plt.xlabel("Feature importance")
        plt.title(f"Feature importance — {target_col}")
        artifact = _save_fig("xgboost_shap_importance.png")
        shap_dict = {k: round(float(v), 4) for k, v in importances.nlargest(10).items()}

    metrics = {
        "model": model_name,
        "target_col": target_col,
        "mae": round(float(mean_absolute_error(y_test, preds)), 4),
        "r2": round(float(r2_score(y_test, preds)), 4),
        "shap_top_features": shap_dict,
        "artifact": artifact,
    }
    print(json.dumps(metrics, indent=2))
    return metrics


# ---------------------------------------------------------------------------
# Registry — this is what the selector agent references in its JSON output
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "linear_regression":            run_linear_regression,
    "random_forest_classifier":     run_random_forest_classifier,
    "kmeans":                       run_kmeans,
    "isolation_forest":             run_isolation_forest,
    "xgboost_shap":                 run_xgboost_shap,
}