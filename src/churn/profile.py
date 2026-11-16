"""One customer, described the way a person would, and translated to the model's columns.

The dashboard collects a :class:`Profile`; :func:`encode` turns it into the one-hot columns
the model was trained on. Keeping the translation here, away from the page code, means it can
be tested without a browser and reused for "what if" comparisons.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

CONTRACTS: tuple[str, ...] = ("Month-to-month", "One year", "Two year")
INTERNET: tuple[str, ...] = ("DSL", "Fiber optic", "No")
PAYMENTS: tuple[str, ...] = (
    "Bank transfer (automatic)",
    "Credit card (automatic)",
    "Electronic check",
    "Mailed check",
)
# Services that only exist on top of an internet connection: column prefix -> label.
ADD_ONS: dict[str, str] = {
    "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection",
    "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
}
# Add-ons a retention offer could plausibly include (streaming is the customer's own choice).
OFFERABLE_ADD_ONS: tuple[str, ...] = (
    "OnlineSecurity",
    "TechSupport",
    "OnlineBackup",
    "DeviceProtection",
)


@dataclass(frozen=True)
class Profile:
    """A customer's account. The defaults describe a typical at-risk customer."""

    contract: str = "Month-to-month"
    tenure: int = 12
    monthly_charges: float = 70.0
    total_charges: float = 840.0
    payment: str = "Electronic check"
    paperless: bool = True
    phone: bool = True
    multiple_lines: bool = False
    internet: str = "Fiber optic"
    add_ons: frozenset[str] = frozenset()
    male: bool = False
    senior: bool = False
    partner: bool = False
    dependents: bool = False


def encode(profile: Profile) -> dict[str, float]:
    """The profile as the model's feature columns (see ``data_prep.encode_features``).

    Impossible combinations can't be expressed: without phone service there are no multiple
    lines, and without internet every add-on is "No internet service".
    """
    has_internet = profile.internet != "No"
    row: dict[str, float] = {
        "SeniorCitizen": int(profile.senior),
        "tenure": profile.tenure,
        "MonthlyCharges": profile.monthly_charges,
        "TotalCharges": profile.total_charges,
        "gender_Male": int(profile.male),
        "Partner_Yes": int(profile.partner),
        "Dependents_Yes": int(profile.dependents),
        "PhoneService_Yes": int(profile.phone),
        "MultipleLines_No phone service": int(not profile.phone),
        "MultipleLines_Yes": int(profile.phone and profile.multiple_lines),
        "InternetService_Fiber optic": int(profile.internet == "Fiber optic"),
        "InternetService_No": int(not has_internet),
        "Contract_One year": int(profile.contract == "One year"),
        "Contract_Two year": int(profile.contract == "Two year"),
        "PaperlessBilling_Yes": int(profile.paperless),
    }
    for add_on in ADD_ONS:
        row[f"{add_on}_No internet service"] = int(not has_internet)
        row[f"{add_on}_Yes"] = int(has_internet and add_on in profile.add_ons)
    for payment in PAYMENTS[1:]:  # the first payment method is the baseline (all zeros)
        row[f"PaymentMethod_{payment}"] = int(profile.payment == payment)
    return row


def alternatives(profile: Profile) -> list[tuple[str, Profile]]:
    """The same customer with one thing changed that a retention team could offer.

    Returns (description, changed profile) pairs: a longer contract, an add-on the customer
    doesn't have yet, or an automatic payment method.
    """
    options: list[tuple[str, Profile]] = []
    current = CONTRACTS.index(profile.contract)
    for contract in CONTRACTS[current + 1 :]:
        label = f"On a {contract.lower().replace(' ', '-')} contract"
        options.append((label, replace(profile, contract=contract)))
    if profile.internet != "No":
        for add_on in OFFERABLE_ADD_ONS:
            if add_on not in profile.add_ons:
                changed = replace(profile, add_ons=profile.add_ons | {add_on})
                options.append((f"With {ADD_ONS[add_on].lower()}", changed))
    if "automatic" not in profile.payment:
        for payment in PAYMENTS[:2]:
            how = payment.split(" (")[0].lower()
            options.append((f"Paying by automatic {how}", replace(profile, payment=payment)))
    return options


def pretty_feature(name: str) -> str:
    """A model column as words: ``Contract_Two year`` -> ``Contract: two year``."""
    head, _, value = name.partition("_")
    words = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", head).split()
    label = " ".join(w if w.isupper() else w.lower() for w in words)
    label = label[:1].upper() + label[1:]
    if not value:
        return label
    value = value if value.split()[0].isupper() else value[:1].lower() + value[1:]
    return f"{label}: {value}"
