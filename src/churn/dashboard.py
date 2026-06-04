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


# Plain-language tooltips shown on each sidebar field.
HELP: dict[str, str] = {
    "gender": "The customer's gender.",
    "SeniorCitizen": "Is the customer 65 or older?",
    "Partner": "Does the customer have a spouse or partner?",
    "Dependents": "Does the customer support dependents (children/family)?",
    "PhoneService": "Does the customer have a phone line with us?",
    "MultipleLines": "Does the customer have more than one phone line?",
    "InternetService": "Type of internet the customer has, or none.",
    "OnlineSecurity": "Optional add-on: online security software.",
    "OnlineBackup": "Optional add-on: cloud backup.",
    "DeviceProtection": "Optional add-on: device protection plan.",
    "TechSupport": "Optional add-on: premium tech support.",
    "StreamingTV": "Does the customer stream TV through us?",
    "StreamingMovies": "Does the customer stream movies through us?",
    "tenure": "Months the customer has been with us. Short tenure is one of "
    "the strongest churn signals.",
    "MonthlyCharges": "What the customer pays per month, in dollars.",
    "TotalCharges": "What the customer has paid in total, in dollars.",
    "Contract": "Contract length. Month-to-month customers churn far more than "
    "those on 1- or 2-year contracts.",
    "PaperlessBilling": "Is the customer on paperless billing?",
    "PaymentMethod": "How the customer pays their bill.",
}

# Default 'average' customer used to seed the sidebar on first load.
DEFAULTS: dict[str, Any] = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
    "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
    "tenure": 12, "MonthlyCharges": 50.0, "TotalCharges": 500.0,
    "Contract": "Month-to-month", "PaperlessBilling": "No",
    "PaymentMethod": "Mailed check",
}

# One-click example profiles so users can see the model work instantly.
PRESETS: dict[str, dict[str, Any]] = {
    "⚠️ High-risk customer": {
        "gender": "Female", "SeniorCitizen": 1, "Partner": "No", "Dependents": "No",
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "Yes",
        "tenure": 2, "MonthlyCharges": 95.0, "TotalCharges": 190.0,
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
    },
    "✅ Loyal customer": {
        "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
        "PhoneService": "Yes", "MultipleLines": "Yes", "InternetService": "DSL",
        "OnlineSecurity": "Yes", "OnlineBackup": "Yes", "DeviceProtection": "Yes",
        "TechSupport": "Yes", "StreamingTV": "No", "StreamingMovies": "No",
        "tenure": 65, "MonthlyCharges": 60.0, "TotalCharges": 3900.0,
        "Contract": "Two year", "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
    },
}

# Internet add-on fields, each encoded as {field}_No internet service / {field}_Yes.
ADDON_FIELDS: list[str] = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]


def encode_inputs(sel: dict[str, Any], feature_names: list[str]) -> dict[str, Any]:
    """Convert human-readable customer selections into the model's one-hot
    feature vector. Pure function (no Streamlit) so it can be unit-tested.
    """
    inputs: dict[str, Any] = {col: 0 for col in feature_names}

    inputs["gender_Male"] = 1 if sel["gender"] == "Male" else 0
    inputs["SeniorCitizen"] = sel["SeniorCitizen"]
    inputs["Partner_Yes"] = 1 if sel["Partner"] == "Yes" else 0
    inputs["Dependents_Yes"] = 1 if sel["Dependents"] == "Yes" else 0

    inputs["PhoneService_Yes"] = 1 if sel["PhoneService"] == "Yes" else 0
    inputs["MultipleLines_No phone service"] = (
        1 if sel["MultipleLines"] == "No phone service" else 0
    )
    inputs["MultipleLines_Yes"] = 1 if sel["MultipleLines"] == "Yes" else 0

    inputs["InternetService_Fiber optic"] = (
        1 if sel["InternetService"] == "Fiber optic" else 0
    )
    inputs["InternetService_No"] = 1 if sel["InternetService"] == "No" else 0

    for field in ADDON_FIELDS:
        inputs[f"{field}_No internet service"] = (
            1 if sel[field] == "No internet service" else 0
        )
        inputs[f"{field}_Yes"] = 1 if sel[field] == "Yes" else 0

    inputs["tenure"] = sel["tenure"]
    inputs["MonthlyCharges"] = sel["MonthlyCharges"]
    inputs["TotalCharges"] = sel["TotalCharges"]

    inputs["Contract_One year"] = 1 if sel["Contract"] == "One year" else 0
    inputs["Contract_Two year"] = 1 if sel["Contract"] == "Two year" else 0
    inputs["PaperlessBilling_Yes"] = 1 if sel["PaperlessBilling"] == "Yes" else 0

    pay = sel["PaymentMethod"]
    inputs["PaymentMethod_Credit card (automatic)"] = (
        1 if pay == "Credit card (automatic)" else 0
    )
    inputs["PaymentMethod_Electronic check"] = 1 if pay == "Electronic check" else 0
    inputs["PaymentMethod_Mailed check"] = 1 if pay == "Mailed check" else 0

    return inputs


def _apply_preset(name: str) -> None:
    """Button callback: load a preset profile into the sidebar widgets."""
    for key, value in PRESETS[name].items():
        st.session_state[key] = value
    st.session_state["show_prediction"] = True


def get_sidebar_inputs(feature_names: list[str]) -> dict[str, Any]:
    """Render the sidebar and return the encoded feature vector.

    Multiple Lines and internet add-ons are auto-resolved and disabled when
    they don't apply (no phone / no internet), so inputs stay self-consistent.
    """
    # Seed defaults once; widgets and presets then share this session state.
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)

    # ---- Example presets ----
    st.sidebar.caption("New here? Load an example, then click Predict.")
    for name in PRESETS:
        st.sidebar.button(
            name, on_click=_apply_preset, args=(name,), use_container_width=True
        )

    sel: dict[str, Any] = {}

    # ---- Demographics ----
    st.sidebar.subheader("Demographics")
    sel["gender"] = st.sidebar.selectbox(
        "Gender", ["Female", "Male"], key="gender", help=HELP["gender"]
    )
    sel["SeniorCitizen"] = st.sidebar.selectbox(
        "Senior Citizen", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes",
        key="SeniorCitizen", help=HELP["SeniorCitizen"],
    )
    sel["Partner"] = st.sidebar.selectbox(
        "Partner", ["No", "Yes"], key="Partner", help=HELP["Partner"]
    )
    sel["Dependents"] = st.sidebar.selectbox(
        "Dependents", ["No", "Yes"], key="Dependents", help=HELP["Dependents"]
    )

    # ---- Services ----
    st.sidebar.subheader("Services")
    sel["PhoneService"] = st.sidebar.selectbox(
        "Phone Service", ["No", "Yes"], key="PhoneService", help=HELP["PhoneService"]
    )
    has_phone = sel["PhoneService"] == "Yes"
    ml_choice = st.sidebar.selectbox(
        "Multiple Lines", ["No", "Yes"], key="MultipleLines",
        disabled=not has_phone, help=HELP["MultipleLines"],
    )
    sel["MultipleLines"] = ml_choice if has_phone else "No phone service"
    if not has_phone:
        st.sidebar.caption("↳ No phone service — multiple lines not applicable.")

    sel["InternetService"] = st.sidebar.selectbox(
        "Internet Service", ["DSL", "Fiber optic", "No"],
        key="InternetService", help=HELP["InternetService"],
    )
    has_net = sel["InternetService"] != "No"
    for field, label in [
        ("OnlineSecurity", "Online Security"),
        ("OnlineBackup", "Online Backup"),
        ("DeviceProtection", "Device Protection"),
        ("TechSupport", "Tech Support"),
        ("StreamingTV", "Streaming TV"),
        ("StreamingMovies", "Streaming Movies"),
    ]:
        choice = st.sidebar.selectbox(
            label, ["No", "Yes"], key=field, disabled=not has_net, help=HELP[field]
        )
        sel[field] = choice if has_net else "No internet service"
    if not has_net:
        st.sidebar.caption("↳ No internet service — add-ons above not applicable.")

    # ---- Account Info ----
    st.sidebar.subheader("Account Info")
    sel["tenure"] = st.sidebar.number_input(
        "Tenure (months)", min_value=0, max_value=72, step=1,
        key="tenure", help=HELP["tenure"],
    )
    sel["MonthlyCharges"] = st.sidebar.number_input(
        "Monthly Charges ($)", min_value=0.0, max_value=120.0, step=1.0,
        key="MonthlyCharges", help=HELP["MonthlyCharges"],
    )
    sel["TotalCharges"] = st.sidebar.number_input(
        "Total Charges ($)", min_value=0.0, max_value=10000.0, step=10.0,
        key="TotalCharges", help=HELP["TotalCharges"],
    )
    sel["Contract"] = st.sidebar.selectbox(
        "Contract", ["Month-to-month", "One year", "Two year"],
        key="Contract", help=HELP["Contract"],
    )
    sel["PaperlessBilling"] = st.sidebar.selectbox(
        "Paperless Billing", ["No", "Yes"],
        key="PaperlessBilling", help=HELP["PaperlessBilling"],
    )
    sel["PaymentMethod"] = st.sidebar.selectbox(
        "Payment Method",
        [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check",
        ],
        key="PaymentMethod", help=HELP["PaymentMethod"],
    )

    return encode_inputs(sel, feature_names)


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
    if st.sidebar.button("Predict", type="primary", use_container_width=True):
        st.session_state["show_prediction"] = True
    # Set by the Predict button and by preset buttons (via _apply_preset).
    show_prediction = st.session_state.get("show_prediction", False)

    # ---- MAIN: tabs ----
    tab1, tab2, tab3 = st.tabs(
        ["Prediction Results", "Model Performance", "Data Insights"]
    )

    with tab1:
        st.subheader("Prediction Results")
        if show_prediction:
            row_df = build_input_row(feature_names, inputs)
            proba = float(model.predict_proba(row_df)[0, 1])
            proba_pct = round(proba * 100, 1)
            risk, risk_color = risk_level(proba)

            # ---- Plain-English verdict (the headline result) ----
            verdict = (
                f"This customer has a **{proba_pct:.1f}% chance of churning** "
                f"— **{risk.lower()} risk**."
            )
            if risk == "High":
                st.error(f"🔴 {verdict}  Consider a retention offer.")
            elif risk == "Medium":
                st.warning(f"🟠 {verdict}  Worth keeping an eye on.")
            else:
                st.success(f"🟢 {verdict}  This customer looks likely to stay.")

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

            st.markdown("**Top 3 factors the model weighs most overall:**")
            for i, (name, importance) in enumerate(
                top_factors(model.feature_importances_, feature_names), 1
            ):
                st.write(f"{i}. {name} (importance: {importance:.3f})")
        else:
            st.info(
                "Set the customer's details in the sidebar (or load an example) "
                "and click **Predict** to see their churn probability and risk level."
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
