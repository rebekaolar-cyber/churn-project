"""Tests for the dashboard's HTML blocks: right numbers, right emphasis, safe text."""

from __future__ import annotations

import plotly.graph_objects as go
import pytest

from churn import ui


def test_masthead_states_the_three_facts():
    html = ui.masthead(customers=7043, churn_rate=0.2654, accuracy=0.7807, tested=1409)
    assert "7,043" in html and "26.5%" in html and "78.1%" in html
    assert "1,409 customers it never saw" in html


@pytest.mark.parametrize(
    "proba,risk,css,band",
    [
        (0.12, "Low", "ch-low", "under 33%"),
        (0.5, "Medium", "ch-medium", "33% to 66%"),
        (0.84, "High", "ch-high", "66% or more"),
    ],
)
def test_verdict_names_the_risk_in_words_as_well_as_colour(proba, risk, css, band):
    html = ui.verdict(proba, risk, bins=[10, 5, 3, 2], below=0.6, tested=20)
    assert css in html
    assert f"{risk} risk" in html and band in html
    assert f"{proba * 100:.0f}<span>%</span>" in html
    assert "Higher than <b>60%</b> of the 20 customers" in html


def test_verdict_bars_share_one_scale_and_mark_the_customers_position():
    html = ui.verdict(0.5, "Medium", bins=[8, 4, 2, 0], below=0.7, tested=14)
    assert 'style="height:100.0%"' in html and 'style="height:50.0%"' in html
    # Two of the four slices lie at or below 50%: those bars are drawn solid.
    assert html.count('class="ch-below"') == 2
    assert 'style="left:50.0%"' in html


def test_what_ifs_list_each_change_with_its_drop():
    html = ui.what_ifs(0.84, [("On a one-year contract", 0.33)], ["Total charges", "Tenure"])
    assert "On a one-year contract" in html
    assert "51 points lower" in html and "33%" in html
    assert "leans most on total charges and tenure" in html


def test_what_ifs_say_so_when_nothing_lowers_the_estimate():
    html = ui.what_ifs(0.05, [], ["Tenure"])
    assert "ch-if-name" not in html
    assert "lowers the model's estimate" in html


def test_outcomes_put_each_count_in_its_own_cell():
    html = ui.outcomes(stayed_right=867, stayed_flagged=168, left_missed=141, left_caught=233)
    for count, meaning in ((867, "left alone"), (168, "flagged, but stayed"), (141, "missed")):
        assert f"<b>{count}</b><span>{meaning}" in html
    assert "<b>233</b><span>caught before leaving" in html


def test_split_and_rates_are_in_proportion():
    html = ui.split(stayed=75, left=25)
    assert "75.0% stayed" in html and "25.0% left" in html
    assert 'style="flex:75"' in html and 'style="flex:25"' in html

    html = ui.rates([("Month-to-month contract", 40, 100), ("Two year contract", 10, 100)])
    assert "40 of 100 left" in html and ">40%<" in html and ">10%<" in html
    assert "width:100.0%" in html and "width:25.0%" in html  # longest bar is the full width


def test_text_is_escaped():
    assert "<script>" not in ui.group("<script>")
    assert "<script>" not in ui.readings([("<script>", "<script>")])
    assert "<script>" not in ui.what_ifs(0.5, [("<script>", 0.2)], ["<script>"])
    assert "<script>" not in ui.rates([("<script>", 1, 2)])
    assert "<script>" not in ui.footer('"><script>')


def test_style_removes_chart_chrome():
    fig = ui.style(go.Figure(go.Bar(x=[1], y=[1])), height=300)
    assert fig.layout.height == 300
    assert fig.layout.showlegend is False
    assert fig.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert fig.layout.font.family == ui.FONT
