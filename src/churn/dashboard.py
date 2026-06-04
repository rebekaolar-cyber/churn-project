"""Customer Churn Prediction Dashboard - Streamlit app.

Header: title, description, key metrics.
Sidebar: input features (Demographics, Services, Account Info) + Predict button.
Main: tabs for Prediction Results, Model Performance, Data Insights.
"""

from __future__ import annotations

from typing import Any, cast

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    roc_curve,
)

from churn import config


@st.cache_resource
def load_model_and_artifacts() -> tuple[Any | None, list[str] | None]:
    """Load the trained model and feature list once (None if not yet trained)."""
    if not config.MODEL_FILE.exists() or not config.FEATURES_FILE.exists():
        return None, None
    model = joblib.load(config.MODEL_FILE)
    feature_names = joblib.load(config.FEATURES_FILE)
    return model, feature_names


@st.cache_data
def load_processed_data() -> (
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]
):
    """Load processed train/test splits and raw data for metrics and EDA."""
    X_train = pd.read_csv(config.X_TRAIN_FILE)
    X_test = pd.read_csv(config.X_TEST_FILE)
    y_train = cast(pd.Series, pd.read_csv(config.Y_TRAIN_FILE).squeeze("columns"))
    y_test = cast(pd.Series, pd.read_csv(config.Y_TEST_FILE).squeeze("columns"))
    df_raw = pd.read_csv(config.RAW_DATA_FILE)
    return X_train, X_test, y_train, y_test, df_raw


def build_input_row(
    feature_names: list[str], inputs: dict[str, Any]
) -> pd.DataFrame:
    """Build a single-row DataFrame with columns in the exact feature order."""
    row = {col: inputs.get(col, 0) for col in feature_names}
    return pd.DataFrame([row])[feature_names]


def risk_level(proba: float) -> tuple[str, str]:
    """Map a churn probability to a (label, color) pair."""
    if proba < 0.33:
        return "Low", "green"
    if proba < 0.66:
        return "Medium", "orange"
    return "High", "red"


def top_factors(
    importances: np.ndarray, feature_names: list[str], n: int = 3
) -> list[tuple[str, float]]:
    """Return the top-n (feature_name, importance) pairs, most important first."""
    idx = np.argsort(importances)[::-1][:n]
    return [(feature_names[i], float(importances[i])) for i in idx]


def get_sidebar_inputs(feature_names: list[str]) -> dict[str, Any]:
    """Collect sidebar inputs and return a feature_name -> value mapping.

    Inputs are grouped: Demographics, Services, Account Info.
    """
    inputs: dict[str, Any] = {col: 0 for col in feature_names}

    # ---- Demographics ----
    st.sidebar.subheader("Demographics")
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    inputs["gender_Male"] = 1 if gender == "Male" else 0
    inputs["SeniorCitizen"] = st.sidebar.selectbox(
        "Senior Citizen", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes"
    )
    partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
    inputs["Partner_Yes"] = 1 if partner == "Yes" else 0
    dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])
    inputs["Dependents_Yes"] = 1 if dependents == "Yes" else 0

    # ---- Services ----
    st.sidebar.subheader("Services")
    phone = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
    inputs["PhoneService_Yes"] = 1 if phone == "Yes" else 0

    mult_lines = st.sidebar.selectbox(
        "Multiple Lines", ["No phone service", "No", "Yes"]
    )
    inputs["MultipleLines_No phone service"] = 1 if mult_lines == "No phone service" else 0
    inputs["MultipleLines_Yes"] = 1 if mult_lines == "Yes" else 0

    internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    inputs["InternetService_Fiber optic"] = 1 if internet == "Fiber optic" else 0
    inputs["InternetService_No"] = 1 if internet == "No" else 0

    opt_vals = ["No internet service", "No", "Yes"]
    security = st.sidebar.selectbox("Online Security", opt_vals)
    inputs["OnlineSecurity_No internet service"] = 1 if security == "No internet service" else 0
    inputs["OnlineSecurity_Yes"] = 1 if security == "Yes" else 0

    backup = st.sidebar.selectbox("Online Backup", opt_vals)
    inputs["OnlineBackup_No internet service"] = 1 if backup == "No internet service" else 0
    inputs["OnlineBackup_Yes"] = 1 if backup == "Yes" else 0

    device = st.sidebar.selectbox("Device Protection", opt_vals)
    inputs["DeviceProtection_No internet service"] = 1 if device == "No internet service" else 0
    inputs["DeviceProtection_Yes"] = 1 if device == "Yes" else 0

    tech = st.sidebar.selectbox("Tech Support", opt_vals)
    inputs["TechSupport_No internet service"] = 1 if tech == "No internet service" else 0
    inputs["TechSupport_Yes"] = 1 if tech == "Yes" else 0

    stream_tv = st.sidebar.selectbox("Streaming TV", opt_vals)
    inputs["StreamingTV_No internet service"] = 1 if stream_tv == "No internet service" else 0
    inputs["StreamingTV_Yes"] = 1 if stream_tv == "Yes" else 0

    stream_mov = st.sidebar.selectbox("Streaming Movies", opt_vals)
    inputs["StreamingMovies_No internet service"] = 1 if stream_mov == "No internet service" else 0
    inputs["StreamingMovies_Yes"] = 1 if stream_mov == "Yes" else 0

    # ---- Account Info ----
    st.sidebar.subheader("Account Info")
    inputs["tenure"] = st.sidebar.number_input(
        "Tenure (months)", min_value=0, max_value=72, value=12, step=1
    )
    inputs["MonthlyCharges"] = st.sidebar.number_input(
        "Monthly Charges ($)", min_value=0.0, max_value=120.0, value=50.0, step=1.0
    )
    inputs["TotalCharges"] = st.sidebar.number_input(
        "Total Charges ($)", min_value=0.0, max_value=10000.0, value=500.0, step=10.0
    )

    contract = st.sidebar.selectbox(
        "Contract", ["Month-to-month", "One year", "Two year"]
    )
    inputs["Contract_One year"] = 1 if contract == "One year" else 0
    inputs["Contract_Two year"] = 1 if contract == "Two year" else 0

    paperless = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
    inputs["PaperlessBilling_Yes"] = 1 if paperless == "Yes" else 0

    payment = st.sidebar.selectbox(
        "Payment Method",
        [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check",
        ],
    )
    cc_auto = "Credit card (automatic)"
    inputs[f"PaymentMethod_{cc_auto}"] = 1 if payment == cc_auto else 0
    inputs["PaymentMethod_Electronic check"] = 1 if payment == "Electronic check" else 0
    inputs["PaymentMethod_Mailed check"] = 1 if payment == "Mailed check" else 0

    return inputs


def inject_custom_css() -> None:
    """Inject custom CSS for nicer dashboard styling."""
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%); }
        h1 {
            color: #1e293b !important; font-weight: 700 !important;
            padding-bottom: 0.25rem; border-bottom: 3px solid #3b82f6;
            margin-bottom: 0.5rem !important;
        }
        h2, h3 { color: #334155 !important; font-weight: 600 !important; }
        [data-testid="stMetric"] {
            background: white; padding: 1rem 1.25rem; border-radius: 10px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08); border: 1px solid #e2e8f0;
        }
        [data-testid="stMetricLabel"] { color: #64748b !important; font-weight: 500 !important; }
        [data-testid="stMetricValue"] { color: #1e293b !important; font-weight: 700 !important; }
        [data-testid="stSidebar"] { background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%); }
        [data-testid="stSidebar"] .stMarkdown { color: #f1f5f9; }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: #f8fafc !important;
        }
        [data-testid="stSidebar"] label { color: #cbd5e1 !important; }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px; background: #e2e8f0; padding: 6px; border-radius: 10px;
        }
        .stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 10px 20px; font-weight: 500; }
        .stTabs [aria-selected="true"] { background: #3b82f6 !important; color: white !important; }
        hr { margin: 1.5rem 0 !important; border-color: #e2e8f0 !important; }
        .stMarkdown p { color: #475569; line-height: 1.6; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Customer Churn Prediction Dashboard",
        page_icon="📊",
        layout="wide",
    )
    inject_custom_css()

    model, feature_names = load_model_and_artifacts()
    if model is None or feature_names is None:
        st.error(
            "Model or features not found. Train first with "
            "`python -m churn.train` so that `models/` contains "
            "rf_model.joblib and features.joblib."
        )
        return

    # ---- HEADER ----
    st.title("Customer Churn Prediction Dashboard")
    st.markdown(
        "This dashboard predicts customer churn for a telecom dataset. "
        "Use the sidebar to set customer features and click **Predict** to see the "
        "churn probability, risk level, and top influencing factors. "
        "You can also explore model performance and data insights in the tabs below."
    )

    X_train, X_test, y_train, y_test, df_raw = load_processed_data()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    total_customers = len(df_raw)
    churn_rate = (
        (df_raw[config.TARGET_COLUMN] == "Yes").mean()
        if config.TARGET_COLUMN in df_raw.columns
        else 0
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Churn Rate", f"{churn_rate:.1%}")
    col3.metric("Model Accuracy", f"{accuracy:.1%}")

    st.divider()

    # ---- SIDEBAR: input features + predict ----
    st.sidebar.header("Input Features")
    inputs = get_sidebar_inputs(feature_names)
    predict_clicked = st.sidebar.button("Predict", type="primary")

    # ---- MAIN: tabs ----
    tab1, tab2, tab3 = st.tabs(
        ["Prediction Results", "Model Performance", "Data Insights"]
    )

    with tab1:
        st.subheader("Prediction Results")
        if predict_clicked:
            row_df = build_input_row(feature_names, inputs)
            proba = float(model.predict_proba(row_df)[0, 1])
            proba_pct = round(proba * 100, 1)

            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=proba_pct,
                    number={"suffix": "%", "font": {"size": 40}},
                    title={"text": "Churn Probability"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 33], "color": "lightgreen"},
                            {"range": [33, 66], "color": "yellow"},
                            {"range": [66, 100], "color": "lightcoral"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "value": 50,
                        },
                    },
                )
            )
            fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            risk, risk_color = risk_level(proba)
            st.markdown(f"**Risk level:** :{risk_color}[**{risk}**]")

            st.markdown("**Top 3 factors influencing this prediction:**")
            for i, (name, importance) in enumerate(
                top_factors(model.feature_importances_, feature_names), 1
            ):
                st.write(f"{i}. {name} (importance: {importance:.3f})")
        else:
            st.info(
                "Click **Predict** in the sidebar to see the churn probability "
                "and risk level."
            )

    with tab2:
        st.subheader("Model Performance")

        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax, cmap="Blues")
        ax.set_title("Confusion Matrix (Test Set)")
        st.pyplot(fig_cm)
        plt.close()

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        fig_roc = go.Figure()
        fig_roc.add_trace(
            go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"ROC (AUC = {roc_auc:.3f})",
                line=dict(color="blue", width=2),
            )
        )
        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Random",
                line=dict(color="gray", dash="dash"),
            )
        )
        fig_roc.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=400,
            showlegend=True,
        )
        st.plotly_chart(fig_roc, use_container_width=True)

        imp_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": model.feature_importances_}
        ).sort_values("Importance", ascending=True)
        fig_imp = px.bar(
            imp_df.tail(15), x="Importance", y="Feature", orientation="h",
            title="Feature Importance (Top 15)",
        )
        fig_imp.update_layout(height=500, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_imp, use_container_width=True)

    with tab3:
        st.subheader("Data Insights")
        if config.TARGET_COLUMN in df_raw.columns:
            churn_counts = df_raw[config.TARGET_COLUMN].value_counts().reset_index()
            churn_counts.columns = ["Churn", "Count"]
            fig_churn = px.bar(
                churn_counts, x="Churn", y="Count",
                title="Distribution of Churn", color="Churn",
            )
            st.plotly_chart(fig_churn, use_container_width=True)

        if "tenure" in df_raw.columns and config.TARGET_COLUMN in df_raw.columns:
            fig_tenure = px.box(
                df_raw, x="Churn", y="tenure",
                title="Tenure by Churn Status", color="Churn",
            )
            st.plotly_chart(fig_tenure, use_container_width=True)

        if "MonthlyCharges" in df_raw.columns and config.TARGET_COLUMN in df_raw.columns:
            fig_mc = px.box(
                df_raw, x="Churn", y="MonthlyCharges",
                title="Monthly Charges by Churn Status", color="Churn",
            )
            st.plotly_chart(fig_mc, use_container_width=True)

        if "Contract" in df_raw.columns and config.TARGET_COLUMN in df_raw.columns:
            ct = df_raw.groupby(["Contract", "Churn"]).size().reset_index(name="Count")
            fig_contract = px.bar(
                ct, x="Contract", y="Count", color="Churn", barmode="group",
                title="Churn by Contract Type",
            )
            st.plotly_chart(fig_contract, use_container_width=True)


if __name__ == "__main__":
    main()
