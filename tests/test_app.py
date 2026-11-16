"""The whole dashboard runs: it builds its model if needed and renders without an exception."""

from __future__ import annotations

from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"


def test_the_dashboard_runs_with_every_input_on_the_page():
    # One run only: this Streamlit version's test harness can't re-send a single-choice
    # segmented control, so a second run would fail in the harness, not in the app.
    at = AppTest.from_file(str(APP), default_timeout=300).run()
    assert not at.exception, [e.value for e in at.exception]
    assert [tab.label for tab in at.tabs] == [
        "One customer",
        "How good is the model?",
        "What the data says",
    ]
    labels = {widget.label for widget in [*at.slider, *at.toggle, *at.selectbox, *at.number_input]}
    assert {
        "Months with the company",
        "Monthly charges",
        "Total charges so far ($)",
        "Payment method",
        "Paperless billing",
        "Phone service",
        "More than one phone line",
        "Senior citizen",
        "Has a partner",
        "Has dependents",
    } <= labels
