"""The dashboard's look: colours, page styles, HTML blocks and the chart style.

Nothing here touches the model. Each block takes numbers the dashboard already has and
returns HTML for ``st.html``; ``style`` dresses a Plotly figure. Colours match
``.streamlit/config.toml``.

Colour means one thing everywhere: red is leaving (customers who left, high risk), blue is
staying (customers who stayed, low risk), amber is in between. Every coloured figure also
says in words what it is, so colour is never the only signal.
"""

from __future__ import annotations

from html import escape
from typing import Any

INK, INK2, MUTED, LINE = "#101218", "#4A4F5C", "#646A78", "#DADDE3"
PAGE, CARD = "#F3F4F6", "#FFFFFF"
STAYS, LEAVES, BETWEEN = "#2447F0", "#D93A26", "#FFB703"
RISK = {"Low": "low", "Medium": "medium", "High": "high"}  # risk label -> CSS class
FONT = "Archivo, system-ui, sans-serif"

CSS = f"""
<style>
:root {{
  --ch-ink:{INK}; --ch-ink2:{INK2}; --ch-muted:{MUTED}; --ch-line:{LINE};
  --ch-page:{PAGE}; --ch-card:{CARD};
  --ch-stays:{STAYS}; --ch-leaves:{LEAVES}; --ch-between:{BETWEEN};
  --ch-wide:"wdth" 125;
}}
[data-testid="stHeader"] {{ background:transparent; }}
[data-testid="stMainBlockContainer"] {{
  padding-top:3.25rem; padding-bottom:4rem; max-width:90rem;
}}
h2, h3 {{ font-variation-settings:"wdth" 112; letter-spacing:-0.01em; }}

/* ---- masthead: the question, and the facts of the data set as a short spec sheet */
.ch-mast {{
  display:grid; grid-template-columns:minmax(0,1.45fr) minmax(0,1fr); gap:1.5rem 4rem;
  align-items:end; padding-bottom:2.2rem; margin-bottom:.4rem;
  border-bottom:3px solid var(--ch-ink);
}}
.ch-mast h1 {{
  font-weight:900; font-variation-settings:var(--ch-wide); color:var(--ch-ink);
  font-size:clamp(2.3rem, 5.1vw, 4.6rem); line-height:.95; letter-spacing:-0.025em;
  margin:0; padding:0; text-wrap:balance;
}}
.ch-mast p {{
  margin:0 0 1rem; color:var(--ch-ink2); font-size:1.02rem; line-height:1.5; max-width:34em;
}}
.ch-facts {{ margin:0 !important; padding:0 !important; }}
.ch-facts div {{
  display:flex; justify-content:space-between; align-items:baseline; gap:1rem;
  padding:.45rem 0; border-top:1px solid var(--ch-line);
}}
.ch-facts dt {{ color:var(--ch-ink2); font-size:.95rem; }}
.ch-facts dd {{
  margin:0; font-weight:800; font-variation-settings:var(--ch-wide); font-size:1.3rem;
  color:var(--ch-ink); white-space:nowrap;
}}
@media (max-width: 900px) {{ .ch-mast {{ grid-template-columns:minmax(0,1fr); }} }}

/* ---- tabs: plain words with a heavy rule under the open one */
.stTabs [data-baseweb="tab-list"] {{ gap:2.2rem; }}
.stTabs [data-baseweb="tab"] {{ height:auto; padding:.9rem 0 .75rem; background:transparent; }}
.stTabs [data-baseweb="tab"] p {{ font-size:1.05rem; font-weight:600; color:var(--ch-muted); }}
.stTabs [data-baseweb="tab"][aria-selected="true"] p {{ color:var(--ch-ink); }}
.stTabs [data-baseweb="tab"]:hover p {{ color:var(--ch-ink); }}
.stTabs [data-baseweb="tab-highlight"] {{ background:var(--ch-ink); height:3px; }}
.stTabs [data-baseweb="tab-border"] {{ background:var(--ch-line); }}
.stTabs [data-baseweb="tab-panel"] {{ padding-top:1.6rem; }}

/* ---- the form's group names */
.ch-group {{
  font-weight:800; font-variation-settings:"wdth" 112; font-size:1.05rem; color:var(--ch-ink);
  margin:.4rem 0 0; padding:.9rem 0 0; border-top:1px solid var(--ch-line);
}}
.ch-group.ch-first {{ border-top:0; padding-top:0; margin-top:0; }}

/* ---- the verdict: one colour field that turns from blue to amber to red with the risk */
[data-testid="stColumn"]:has(.st-key-verdict) {{
  position:sticky; top:1.25rem; align-self:flex-start;
}}
.ch-verdict {{ padding:1.6rem 1.8rem 1.5rem; color:#fff; background:var(--ch-stays); }}
.ch-verdict.ch-medium {{ background:var(--ch-between); color:var(--ch-ink); }}
.ch-verdict.ch-high {{ background:var(--ch-leaves); }}
.ch-verdict p {{ margin:0; color:inherit; }}
.ch-level {{
  font-weight:700; font-size:1rem; display:flex; justify-content:space-between; gap:1rem;
}}
.ch-level span {{ font-weight:500; }}
.ch-pct {{
  font-weight:900; font-variation-settings:var(--ch-wide); letter-spacing:-0.04em;
  font-size:clamp(5rem, 11.5vw, 10rem); line-height:.86; margin:.9rem 0 .2rem -.04em;
}}
.ch-pct span {{ font-size:.42em; letter-spacing:0; margin-left:.06em; vertical-align:.7em; }}
.ch-say {{ font-size:1.25rem; font-weight:600; line-height:1.3; }}
.ch-strip {{
  position:relative; display:flex; align-items:flex-end; gap:2px; height:4.2rem;
  margin-top:1.7rem;
}}
.ch-strip i {{ flex:1; min-height:2px; background:currentColor; opacity:.34; }}
.ch-strip i.ch-below {{ opacity:.95; }}
.ch-pin {{ position:absolute; top:-.55rem; bottom:0; width:0; border-left:2px solid currentColor; }}
.ch-pin::before {{
  content:""; position:absolute; top:-.2rem; left:-.42rem; width:.7rem; height:.7rem;
  border-radius:50%; background:currentColor;
}}
.ch-axis {{
  display:flex; justify-content:space-between; font-size:.8rem; margin-top:.35rem; opacity:.92;
}}
.ch-rank {{ font-size:.98rem; line-height:1.45; margin-top:.8rem !important; }}

/* ---- what would lower it */
.ch-ifs {{ margin-top:.4rem; }}
.ch-ifs h3 {{ font-size:1.25rem; font-weight:800; margin:0 0 .5rem; padding:0; }}
.ch-if {{
  display:grid; grid-template-columns:minmax(0,1fr) 5.5rem 3.4rem; gap:.2rem 1rem;
  align-items:center;
  padding:.6rem 0; border-top:1px solid var(--ch-line);
}}
.ch-if-name {{ color:var(--ch-ink); font-weight:500; }}
.ch-if-name small {{ display:block; color:var(--ch-muted); font-size:.85rem; font-weight:400; }}
.ch-if-bar {{ height:.55rem; background:#E4E6EB; }}
.ch-if-bar b {{ display:block; height:100%; background:var(--ch-ink); }}
.ch-if-val {{
  font-weight:800; font-variation-settings:var(--ch-wide); font-size:1.15rem; text-align:right;
  color:var(--ch-ink);
}}
.ch-note {{ color:var(--ch-muted); font-size:.875rem; line-height:1.45; margin:.7rem 0 0; }}

/* ---- the four outcomes on the test customers */
.ch-cm {{
  display:grid; grid-template-columns:max-content minmax(0,1fr) minmax(0,1fr); gap:.5rem;
  max-width:44rem;
}}
.ch-cm-top {{
  color:var(--ch-ink2); font-size:.9rem; font-weight:600; padding:0 .2rem .1rem; align-self:end;
}}
.ch-cm-side {{
  color:var(--ch-ink2); font-size:.9rem; font-weight:600; align-self:center; padding-right:.6rem;
}}
.ch-cell {{ padding:1rem 1.1rem 1.05rem; background:#E7EAF0; }}
.ch-cell b {{
  display:block; font-weight:900; font-variation-settings:var(--ch-wide); font-size:2.3rem;
  line-height:1; color:var(--ch-ink);
}}
.ch-cell span {{
  display:block; margin-top:.35rem; color:var(--ch-ink2); font-size:.92rem; line-height:1.3;
}}
.ch-cell.ch-right {{ background:#DCE3FF; }}
.ch-cell.ch-caught {{ background:var(--ch-stays); }}
.ch-cell.ch-caught b, .ch-cell.ch-caught span {{ color:#fff; }}
.ch-cell.ch-wrong {{ background:#FBE0DB; }}
.ch-read {{ margin:0 !important; padding:0 !important; list-style:none; }}
.ch-read li {{ margin:0;
  padding:.7rem 0; border-top:1px solid var(--ch-line); color:var(--ch-ink2); line-height:1.45;
}}
.ch-read li:first-child {{ border-top:0; padding-top:0; }}
.ch-read b {{
  display:block; font-weight:900; font-variation-settings:var(--ch-wide); font-size:1.7rem;
  line-height:1.1; color:var(--ch-ink);
}}

/* ---- a rate per group, as bars on one scale */
.ch-rates {{
  display:grid; grid-template-columns:max-content minmax(4rem,1fr) max-content; gap:.75rem 1.2rem;
  align-items:center; max-width:52rem;
}}
.ch-rate-name {{ color:var(--ch-ink); font-weight:500; }}
.ch-rate-name small {{ display:block; color:var(--ch-muted); font-size:.85rem; font-weight:400; }}
.ch-rate-bar {{ height:1.5rem; }}
.ch-rate-bar b {{ display:block; height:100%; min-width:3px; background:var(--ch-leaves); }}
.ch-rate-val {{
  font-weight:900; font-variation-settings:var(--ch-wide); font-size:1.5rem; color:var(--ch-ink);
  text-align:right;
}}

/* ---- one whole split in two: stayed and left */
.ch-split {{ display:flex; height:3.4rem; gap:2px; }}
.ch-split div {{
  display:flex; align-items:center; padding:0 .9rem; color:#fff; font-weight:700;
  white-space:nowrap; min-width:0;
}}
.ch-split .ch-s {{ background:var(--ch-stays); }}
.ch-split .ch-l {{ background:var(--ch-leaves); }}
.ch-split-key {{
  display:flex; justify-content:space-between; color:var(--ch-ink2); font-size:.92rem;
  margin-top:.4rem; gap:1rem;
}}

.ch-foot {{
  border-top:1px solid var(--ch-line); margin-top:2rem; padding-top:1rem; color:var(--ch-ink2);
  font-size:.9rem; line-height:1.5;
}}
.ch-foot a {{ color:var(--ch-stays); text-underline-offset:3px; }}

@media (max-width: 640px) {{
  /* phones: the verdict comes first, the form below it */
  .st-key-customer [data-testid="stHorizontalBlock"]:has(.st-key-verdict) {{
    flex-direction:column-reverse;
  }}
  [data-testid="stColumn"]:has(.st-key-verdict) {{ position:static; }}
  .stTabs [data-baseweb="tab-list"] {{ gap:1.2rem; }}
  .ch-if {{ grid-template-columns:minmax(0,1fr) 3.4rem; }}
  .ch-if-bar {{ display:none; }}
}}
</style>
"""


def masthead(customers: int, churn_rate: float, accuracy: float, tested: int) -> str:
    return f"""
<header class="ch-mast">
  <h1>Who is about to leave?</h1>
  <div>
    <p>A churn model for a telecom company. Describe a customer to see the model's estimate,
    then check how far to trust it.</p>
    <dl class="ch-facts">
      <div><dt>Customers in the data</dt><dd>{customers:,}</dd></div>
      <div><dt>Share who left</dt><dd>{churn_rate:.1%}</dd></div>
      <div><dt>Model right, on {tested:,} customers it never saw</dt><dd>{accuracy:.1%}</dd></div>
    </dl>
  </div>
</header>"""


def group(title: str, first: bool = False) -> str:
    return f'<p class="ch-group{" ch-first" if first else ""}">{escape(title)}</p>'


def verdict(proba: float, risk: str, bins: list[int], below: float, tested: int) -> str:
    """The estimate for one customer, and where it sits among the test customers.

    bins: how many test customers scored in each equal slice of 0-100%.
    below: the share of test customers who scored lower than this one.
    """
    top = max(bins) or 1
    bars = []
    for i, count in enumerate(bins):
        passed = ' class="ch-below"' if (i + 1) / len(bins) <= proba else ""
        bars.append(f'<i{passed} style="height:{count / top * 100:.1f}%"></i>')
    band = {"Low": "under 33%", "Medium": "33% to 66%", "High": "66% or more"}[risk]
    label = (
        f"Scores of the {tested:,} test customers from 0% to 100%. "
        f"This customer scores {proba:.0%}, higher than {below:.0%} of them."
    )
    return f"""
<section class="ch-verdict ch-{RISK[risk]}">
  <p class="ch-level">{escape(risk)} risk <span>{band}</span></p>
  <div class="ch-pct">{proba * 100:.0f}<span>%</span></div>
  <p class="ch-say">chance this customer leaves, by the model's estimate</p>
  <div class="ch-strip" role="img" aria-label="{escape(label)}">{"".join(bars)}
    <b class="ch-pin" style="left:{proba * 100:.1f}%"></b></div>
  <div class="ch-axis" aria-hidden="true"><span>0%</span><span>50%</span><span>100%</span></div>
  <p class="ch-rank">Higher than <b>{below:.0%}</b> of the {tested:,} customers the model was
  tested on. Each bar is a group of those customers, from safest to most at risk.</p>
</section>"""


def what_ifs(proba: float, changes: list[tuple[str, float]], leans_on: list[str]) -> str:
    """What the model would say for the same customer with one thing changed.

    changes: (description, new probability), already limited to changes that lower the score.
    leans_on: the model's most important columns overall, as words.
    """
    if changes:
        rows = "".join(
            f'<div class="ch-if"><span class="ch-if-name">{escape(name)}'
            f"<small>{(proba - new) * 100:.0f} points lower</small></span>"
            f'<span class="ch-if-bar"><b style="width:{new * 100:.0f}%"></b></span>'
            f'<span class="ch-if-val">{new * 100:.0f}%</span></div>'
            for name, new in changes
        )
        note = (
            "The model's estimate for the same customer with one thing changed. It shows what "
            "the model has learned, not what an offer would do."
        )
    else:
        rows = ""
        note = (
            "No longer contract, extra service or payment method lowers the model's estimate "
            "for this customer."
        )
    weighs = ", ".join(leans_on[:-1]) + f" and {leans_on[-1]}" if len(leans_on) > 1 else leans_on[0]
    return (
        f'<div class="ch-ifs"><h3>What would lower it</h3>{rows}'
        f'<p class="ch-note">{note} Across all customers the model leans most on '
        f"{escape(weighs.lower())}.</p></div>"
    )


def outcomes(stayed_right: int, stayed_flagged: int, left_missed: int, left_caught: int) -> str:
    """The confusion matrix in words: what the model said against what happened."""
    return f"""
<div class="ch-cm">
  <div></div><div class="ch-cm-top">The model said: stays</div>
  <div class="ch-cm-top">The model said: leaves</div>
  <div class="ch-cm-side">Stayed</div>
  <div class="ch-cell ch-right"><b>{stayed_right:,}</b><span>left alone, correctly</span></div>
  <div class="ch-cell ch-wrong"><b>{stayed_flagged:,}</b><span>flagged, but stayed</span></div>
  <div class="ch-cm-side">Left</div>
  <div class="ch-cell ch-wrong"><b>{left_missed:,}</b><span>missed</span></div>
  <div class="ch-cell ch-caught"><b>{left_caught:,}</b><span>caught before leaving</span></div>
</div>"""


def readings(items: list[tuple[str, str]]) -> str:
    """Short statements with the figure first: (figure, what it means)."""
    rows = "".join(
        f"<li><b>{escape(figure)}</b>{escape(meaning)}</li>" for figure, meaning in items
    )
    return f'<ul class="ch-read">{rows}</ul>'


def split(stayed: int, left: int) -> str:
    """All customers as one bar, split into those who stayed and those who left."""
    total = stayed + left or 1
    return f"""
<div class="ch-split" role="img" aria-label="{stayed:,} customers stayed and {left:,} left.">
  <div class="ch-s" style="flex:{stayed}">{stayed / total:.1%} stayed</div>
  <div class="ch-l" style="flex:{left}">{left / total:.1%} left</div>
</div>
<div class="ch-split-key"><span>{stayed:,} customers</span><span>{left:,} customers</span></div>"""


def rates(rows: list[tuple[str, int, int]]) -> str:
    """The share who left in each group: (group, how many left, how many in the group)."""
    shares = [left / size if size else 0.0 for _, left, size in rows]
    longest = max(shares) or 1.0
    cells = "".join(
        f'<div class="ch-rate-name">{escape(name)}<small>{left:,} of {size:,} left</small></div>'
        f'<div class="ch-rate-bar"><b style="width:{share / longest * 100:.1f}%"></b></div>'
        f'<div class="ch-rate-val">{share:.0%}</div>'
        for (name, left, size), share in zip(rows, shares, strict=True)
    )
    return f'<div class="ch-rates">{cells}</div>'


def footer(repo: str) -> str:
    return (
        '<footer class="ch-foot">Built by Rebeka Olar on the public Telco Customer Churn '
        "data set. "
        f'<a href="{escape(repo)}">Code and tests on GitHub</a>.</footer>'
    )


def style(fig: Any, height: int, legend: bool = False) -> Any:
    """The shared chart style: the page's font, quiet grid, no frame, no chart title."""
    fig.update_layout(
        height=height,
        font={"family": FONT, "size": 13, "color": INK2},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 8, "r": 16, "t": 12, "b": 8},
        showlegend=legend,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0, "title": None},
        hoverlabel={"font": {"family": FONT, "size": 13}, "bgcolor": CARD, "bordercolor": LINE},
        title=None,
    )
    axis = {
        "gridcolor": LINE,
        "zeroline": False,
        "linecolor": "#C4C8D0",
        "ticks": "",
        "title_font": {"size": 12, "color": INK2},
        "tickfont": {"size": 12, "color": MUTED},
        "automargin": True,
    }
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig
