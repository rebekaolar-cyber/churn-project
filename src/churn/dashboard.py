"""Customer churn dashboard (Streamlit).

Masthead: the question and three facts about the data.
Tab 1, one customer: describe an account, see the model's estimate and what would lower it.
Tab 2, the model: its four outcomes on unseen customers, the ROC curve, what it weighs.
Tab 3, the data: how many left, and how leaving relates to contract, tenure and price.

The look lives in ``churn.ui``; the translation from a described customer to the model's
columns lives in ``churn.profile``. This module only wires them to the page.
"""

from __future__ import annotations

from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve

from churn import config, ui
from churn.profile import (
    ADD_ONS,
    CONTRACTS,
    INTERNET,
    PAYMENTS,
    Profile,
    alternatives,
    encode,
    pretty_feature,
)

REPO = "https://github.com/rebekaolar-cyber/churn-project"
HIDE_TOOLBAR = {"displayModeBar": False}


@st.cache_resource
def load_model_and_artifacts() -> tuple[Any | None, list[str] | None]:
    """Load the trained model and feature list once (None if not yet trained)."""
    if not config.MODEL_FILE.exists() or not config.FEATURES_FILE.exists():
        return None, None
    model = joblib.load(config.MODEL_FILE)
    feature_names = joblib.load(config.FEATURES_FILE)
    return model, feature_names


@st.cache_data
def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Load processed train/test splits and raw data for metrics and EDA."""
    X_train = pd.read_csv(config.X_TRAIN_FILE)
    X_test = pd.read_csv(config.X_TEST_FILE)
    y_train = cast(pd.Series, pd.read_csv(config.Y_TRAIN_FILE).squeeze("columns"))
    y_test = cast(pd.Series, pd.read_csv(config.Y_TEST_FILE).squeeze("columns"))
    df_raw = pd.read_csv(config.RAW_DATA_FILE)
    return X_train, X_test, y_train, y_test, df_raw


def build_input_row(feature_names: list[str], inputs: dict[str, Any]) -> pd.DataFrame:
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


@st.cache_data
def score_test_set() -> tuple[np.ndarray, np.ndarray]:
    """The model's answer and its probability for every test customer, computed once."""
    model, _ = load_model_and_artifacts()
    _, X_test, _, _, _ = load_processed_data()
    assert model is not None, "score_test_set() is only called once the model has loaded"
    return model.predict(X_test), model.predict_proba(X_test)[:, 1]


def score(model: Any, feature_names: list[str], profile: Profile) -> float:
    """The model's churn probability for one described customer."""
    row = build_input_row(feature_names, encode(profile))
    return float(model.predict_proba(row)[0, 1])


def lowering_changes(
    model: Any, feature_names: list[str], profile: Profile, proba: float, n: int = 3
) -> list[tuple[str, float]]:
    """The n single changes that lower the estimate most, as (description, new probability).

    Changes that move the estimate by less than one point are left out.
    """
    scored = [
        (label, score(model, feature_names, changed)) for label, changed in alternatives(profile)
    ]
    lower = [(label, new) for label, new in scored if proba - new >= 0.01]
    return sorted(lower, key=lambda item: item[1])[:n]


def pick(label: str, options: tuple[str, ...], default: str, key: str) -> str:
    """A segmented control that always has a value.

    Clicking the selected option again would clear it; the callback puts it back.
    """
    last = f"{key}_last"

    def keep_one() -> None:
        if st.session_state[key] is None:
            st.session_state[key] = st.session_state.get(last, default)

    value = st.segmented_control(label, options, default=default, key=key, on_change=keep_one)
    value = default if value is None else value
    st.session_state[last] = value
    return str(value)


def customer_form() -> Profile:
    """The account sheet: every input the model uses, in three groups."""
    st.html(ui.group("Contract and billing", first=True))
    contract = pick("Contract", CONTRACTS, "Month-to-month", "contract")
    a, b = st.columns(2)
    tenure = a.slider("Months with the company", 0, 72, 12)
    monthly = b.slider("Monthly charges", 18, 120, 70, format="$%d")
    a, b = st.columns(2)
    total = a.number_input(
        "Total charges so far ($)", min_value=0.0, max_value=10000.0, value=840.0, step=10.0
    )
    payment = b.selectbox("Payment method", PAYMENTS, index=2)
    paperless = st.toggle("Paperless billing", value=True)

    st.html(ui.group("Services"))
    a, b = st.columns(2)
    phone = a.toggle("Phone service", value=True)
    lines = b.toggle("More than one phone line", value=False, disabled=not phone)
    internet = pick("Internet", INTERNET, "Fiber optic", "internet")
    add_ons = st.pills(
        "Extras on top of internet",
        list(ADD_ONS),
        format_func=lambda key: ADD_ONS[key],
        selection_mode="multi",
        disabled=internet == "No",
        help="Only available with an internet connection.",
    )

    st.html(ui.group("About the customer"))
    gender = pick("Gender", ("Female", "Male"), "Female", "gender")
    a, b, c = st.columns(3)
    senior = a.toggle("Senior citizen")
    partner = b.toggle("Has a partner")
    dependents = c.toggle("Has dependents")

    return Profile(
        contract=contract,
        tenure=int(tenure),
        monthly_charges=float(monthly),
        total_charges=float(total),
        payment=str(payment),
        paperless=paperless,
        phone=phone,
        multiple_lines=phone and lines,
        internet=internet,
        add_ons=frozenset(add_ons or []) if internet != "No" else frozenset(),
        male=gender == "Male",
        senior=senior,
        partner=partner,
        dependents=dependents,
    )


def main() -> None:
    st.set_page_config(
        page_title="Customer churn",
        page_icon=str(config.PROJECT_ROOT / "static" / "favicon.png"),
        layout="wide",
    )
    st.html(ui.CSS)

    model, feature_names = load_model_and_artifacts()
    if model is None or feature_names is None:
        st.error(
            "Model or features not found. Train first with "
            "`python -m churn.train` so that `models/` contains "
            "rf_model.joblib and features.joblib."
        )
        return

    _, _, _, y_test, df_raw = load_processed_data()
    y_pred, y_proba = score_test_set()
    accuracy = accuracy_score(y_test, y_pred)
    tested = len(y_test)
    left = df_raw[config.TARGET_COLUMN] == "Yes"
    st.html(ui.masthead(len(df_raw), float(left.mean()), float(accuracy), tested))

    one, quality, data = st.tabs(["One customer", "How good is the model?", "What the data says"])

    # ---- one customer: the account sheet on the left, the verdict on the right
    with one, st.container(key="customer"):
        form, result = st.columns([1.05, 1], gap="large")
        with form:
            profile = customer_form()
        proba = score(model, feature_names, profile)
        risk, _ = risk_level(proba)
        bins = np.histogram(y_proba, bins=25, range=(0.0, 1.0))[0]
        leans_on = [
            pretty_feature(name)
            for name, _ in top_factors(model.feature_importances_, feature_names)
        ]
        with result, st.container(key="verdict"):
            st.html(
                ui.verdict(
                    proba, risk, [int(b) for b in bins], float((y_proba < proba).mean()), tested
                )
            )
            st.html(
                ui.what_ifs(proba, lowering_changes(model, feature_names, profile, proba), leans_on)
            )

    # ---- the model: what it got right and wrong on customers it never saw
    with quality:
        (stayed_right, stayed_flagged), (left_missed, left_caught) = confusion_matrix(
            y_test, y_pred
        )
        flagged = stayed_flagged + left_caught
        leavers = left_missed + left_caught
        always_stays = max(leavers, tested - leavers) / tested
        st.subheader(f"Tested on {tested:,} customers it had never seen")
        grid, reading = st.columns([1.5, 1], gap="large")
        grid.html(ui.outcomes(stayed_right, stayed_flagged, left_missed, left_caught))
        reading.html(
            ui.readings(
                [
                    (
                        f"{left_caught / leavers:.0%} of leavers caught",
                        f"{left_caught:,} of the {leavers:,} customers who left were flagged.",
                    ),
                    (
                        f"{left_caught / flagged:.0%} of flags were right",
                        f"{left_caught:,} of the {flagged:,} customers it flagged really left.",
                    ),
                    (
                        f"{accuracy:.1%} right overall",
                        f"Saying “stays” for everyone would be right {always_stays:.1%} "
                        "of the time, and would catch nobody.",
                    ),
                ]
            )
        )

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        st.subheader("How well it ranks customers")
        st.caption(
            f"ROC curve: as the bar for flagging is lowered, more leavers are caught (up) and "
            f"more loyal customers are flagged by mistake (right). Area under the curve: "
            f"{roc_auc:.2f}; a coin flip scores 0.50."
        )
        fig_roc = go.Figure()
        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="A coin flip",
                line={"color": "#9AA0AC", "dash": "dash", "width": 1.5},
                hoverinfo="skip",
            )
        )
        fig_roc.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name="The model",
                line={"color": ui.STAYS, "width": 2.5},
                hovertemplate="Loyal customers flagged: %{x:.0%}<br>Leavers caught: %{y:.0%}"
                "<extra></extra>",
            )
        )
        fig_roc.add_trace(
            go.Scatter(
                x=[stayed_flagged / (tested - leavers)],
                y=[left_caught / leavers],
                mode="markers+text",
                name="Where it flags today",
                marker={"color": ui.INK, "size": 11, "line": {"color": ui.PAGE, "width": 2}},
                text=["flags at 50% or more"],
                textposition="bottom right",
                textfont={"color": ui.INK, "size": 13},
                hoverinfo="skip",
            )
        )
        fig_roc.update_xaxes(
            title_text="Loyal customers flagged by mistake", tickformat=".0%", range=[0, 1]
        )
        fig_roc.update_yaxes(title_text="Leavers caught", tickformat=".0%", range=[0, 1.02])
        st.plotly_chart(ui.style(fig_roc, 400, legend=True), theme=None, config=HIDE_TOOLBAR)

        st.subheader("What the model weighs most")
        st.caption("The 15 inputs the random forest relies on most, across all customers.")
        importance = (
            pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
            .sort_values("importance")
            .tail(15)
        )
        fig_imp = go.Figure(
            go.Bar(
                x=importance.importance,
                y=[pretty_feature(f) for f in importance.feature],
                orientation="h",
                marker={"color": ui.STAYS},
                width=0.62,
                hovertemplate="%{y}: %{x:.3f}<extra></extra>",
            )
        )
        fig_imp.update_xaxes(title_text="Importance, as a share of the total", tickformat=".0%")
        ui.style(fig_imp, 520)
        fig_imp.update_yaxes(showgrid=False, tickfont={"size": 13, "color": ui.INK2})
        st.plotly_chart(fig_imp, theme=None, config=HIDE_TOOLBAR)

    # ---- the data: who left
    with data:
        status = left.map({True: "Left", False: "Stayed"})
        colours = {"Stayed": ui.STAYS, "Left": ui.LEAVES}
        st.subheader(f"{int(left.sum()):,} of {len(df_raw):,} customers left")
        st.html(ui.split(int((~left).sum()), int(left.sum())))

        st.subheader("Short contracts lose the most customers")
        st.caption("Share of customers on each contract who left.")
        by_contract = left.groupby(df_raw["Contract"]).agg(["sum", "size"]).reindex(CONTRACTS)
        st.html(
            ui.rates(
                [
                    (f"{name} contract", int(row["sum"]), int(row["size"]))
                    for name, row in by_contract.iterrows()
                ]
            )
        )

        st.subheader("Leavers are newer customers, and pay more")
        tenure_col, charges_col = st.columns(2, gap="large")
        for column, field, title, unit in (
            (tenure_col, "tenure", "Months with the company", ""),
            (charges_col, "MonthlyCharges", "Monthly charges", "$"),
        ):
            fig = go.Figure()
            for name in ("Stayed", "Left"):
                fig.add_trace(
                    go.Box(
                        y=df_raw.loc[status == name, field],
                        name=name,
                        boxpoints=False,
                        marker={"color": colours[name]},
                        line={"width": 2},
                        fillcolor="rgba(0,0,0,0)",
                        width=0.45,
                    )
                )
            fig.update_yaxes(title_text=title, tickprefix=unit, rangemode="tozero")
            ui.style(fig, 360)
            fig.update_xaxes(showgrid=False, tickfont={"size": 13, "color": ui.INK2})
            column.plotly_chart(fig, theme=None, config=HIDE_TOOLBAR)
        st.caption(
            "Each box holds the middle half of customers; the line inside is the median, and "
            "the whiskers reach the lowest and highest values."
        )

    st.html(ui.footer(REPO))


if __name__ == "__main__":
    main()
