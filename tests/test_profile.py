"""Tests for the customer profile: the translation to model columns and the what-if options."""

from __future__ import annotations

import pandas as pd
import pytest

from churn import data_prep
from churn.profile import ADD_ONS, PAYMENTS, Profile, alternatives, encode, pretty_feature


def profile_from_raw(row: pd.Series) -> Profile:
    """Describe a raw data-set row as a Profile, the way a person would fill in the form."""
    return Profile(
        contract=row["Contract"],
        tenure=int(row["tenure"]),
        monthly_charges=float(row["MonthlyCharges"]),
        total_charges=float(row["TotalCharges"]),
        payment=row["PaymentMethod"],
        paperless=row["PaperlessBilling"] == "Yes",
        phone=row["PhoneService"] == "Yes",
        multiple_lines=row["MultipleLines"] == "Yes",
        internet=row["InternetService"],
        add_ons=frozenset(name for name in ADD_ONS if row[name] == "Yes"),
        male=row["gender"] == "Male",
        senior=bool(row["SeniorCitizen"]),
        partner=row["Partner"] == "Yes",
        dependents=row["Dependents"] == "Yes",
    )


def test_encode_matches_the_training_pipeline_on_real_customers():
    """The form must feed the model exactly what training fed it.

    Every 20th real customer is described as a Profile and encoded; the result has to equal
    the row the preprocessing pipeline produced for that same customer, column for column.
    """
    cleaned = data_prep.clean_data(data_prep.load_raw_data())
    encoded, _ = data_prep.encode_features(cleaned)
    sample = cleaned.iloc[::20]
    assert len(sample) > 300
    for index, row in sample.iterrows():
        ours = encode(profile_from_raw(row))
        theirs = encoded.loc[index].to_dict()
        assert set(ours) == set(theirs)
        assert ours == pytest.approx(theirs), f"customer at row {index}"


def test_encode_cannot_express_impossible_combinations():
    no_internet = encode(Profile(internet="No", add_ons=frozenset({"TechSupport"})))
    assert no_internet["InternetService_No"] == 1
    assert no_internet["TechSupport_No internet service"] == 1
    assert no_internet["TechSupport_Yes"] == 0  # an add-on needs a connection

    no_phone = encode(Profile(phone=False, multiple_lines=True))
    assert no_phone["MultipleLines_No phone service"] == 1
    assert no_phone["MultipleLines_Yes"] == 0  # more lines need a phone service


def test_encode_uses_the_first_payment_method_as_the_baseline():
    row = encode(Profile(payment=PAYMENTS[0]))
    assert sum(value for name, value in row.items() if name.startswith("PaymentMethod_")) == 0
    row = encode(Profile(payment="Mailed check"))
    assert row["PaymentMethod_Mailed check"] == 1


def test_alternatives_change_exactly_one_thing():
    profile = Profile()  # month-to-month, fiber, no add-ons, electronic check
    options = alternatives(profile)
    labels = [label for label, _ in options]
    assert "On a one-year contract" in labels and "On a two-year contract" in labels
    assert "With tech support" in labels and "With online security" in labels
    assert "Paying by automatic bank transfer" in labels
    for _, changed in options:
        differing = [
            field
            for field in profile.__dataclass_fields__
            if getattr(profile, field) != getattr(changed, field)
        ]
        assert len(differing) == 1


def test_alternatives_never_offer_what_the_customer_already_has():
    settled = Profile(
        contract="Two year",
        internet="No",
        payment="Credit card (automatic)",
    )
    assert alternatives(settled) == []

    one_year = Profile(contract="One year", add_ons=frozenset({"TechSupport"}))
    labels = [label for label, _ in alternatives(one_year)]
    assert "On a one-year contract" not in labels  # no shorter or equal contract
    assert "With tech support" not in labels
    assert "With streaming TV" not in labels  # streaming is the customer's choice, not an offer


@pytest.mark.parametrize(
    "column,words",
    [
        ("TotalCharges", "Total charges"),
        ("tenure", "Tenure"),
        ("SeniorCitizen", "Senior citizen"),
        ("Contract_Two year", "Contract: two year"),
        ("gender_Male", "Gender: male"),
        ("StreamingTV_Yes", "Streaming TV: yes"),
        ("InternetService_Fiber optic", "Internet service: fiber optic"),
        ("MultipleLines_No phone service", "Multiple lines: no phone service"),
        ("PaymentMethod_Credit card (automatic)", "Payment method: credit card (automatic)"),
    ],
)
def test_pretty_feature(column, words):
    assert pretty_feature(column) == words
