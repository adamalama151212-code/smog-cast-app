"""Charts of the interface, drawn by us (HTML bars, one SVG plot) so they follow the design tokens exactly, in both
themes and at phone width. The page state computes the geometry (``*_rows`` / ``calibration_*`` below, plain
Python); the components only draw it.

Rules (dataviz skill): one job per chart; thin marks with rounded ends; values as direct labels; text in ink colours,
never in a series colour; pollutants as aqua / violet with a marker shape, never the forecast-state colours; every
mark carries a native tooltip; the numbers are also in a table on the same page.
"""

from __future__ import annotations

import reflex as rx

from smogcast.app.forecast import POLLUTANT_LABEL, pct

MODEL_LABEL = {"logistic": "logistic regression", "gbt": "GBT (gradient-boosted trees)",
               "persistence": "persistence (“tomorrow like today”)", "climatology": "climatology (calendar)"}
DQ_LABEL = {"silver_valid": "valid (used for forecasts)", "flag_frozen": "flag: frozen reading",
            "flag_spike": "flag: spike", "flag_drift": "flag: drift", "late_rejected": "rejected: late",
            "quarantined_missing_value": "quarantine: missing value", "quarantined_below_min": "quarantine: negative value",
            "quarantined_above_max": "quarantine: > 1000 µg/m³", "quarantined_unknown_station": "quarantine: unknown station"}
SERIES = {"PM10": "pm10", "PM2.5": "pm25"}


# --------------------------------------------------------------------------------------------- bars (HTML)
def brier_rows(metrics: list[dict], pollutant: str, operational: str | None, city_id: str = "ALL") -> list[dict]:
    """Brier score of every model on the test year, best (lowest) first; the operational model highlighted.
    Each pollutant has its own scale: compare models within one pollutant, not PM10 with PM2.5."""
    data = sorted((r for r in metrics if r["pollutant"] == pollutant and r["city_id"] == city_id
                   and r["brier"] is not None), key=lambda r: r["brier"])
    top = max((r["brier"] for r in data), default=0) or 1
    out = []
    for r in data:
        used = r["model"] == operational
        bss = "" if r.get("bss") is None else f", BSS {r['bss']:.2f}"
        out.append({"label": MODEL_LABEL.get(r["model"], r["model"]) + (" ★" if used else ""),
                    "value": f"{r['brier']:.3f}", "w": f"{r['brier'] / top * 100:.1f}%",
                    "cls": "bar-row " + (SERIES[pollutant] if used else "muted"),
                    "title": f"{MODEL_LABEL.get(r['model'], r['model'])}: Brier {r['brier']:.4f}{bss}"})
    return out


def dq_rows(rows: list[dict], source: str) -> list[dict]:
    """What happened to the unique readings of one source, as shares of them; valid readings highlighted."""
    data = sorted((r for r in rows if r["source"] == source and r["metric"] in DQ_LABEL),
                  key=lambda r: -(r["pct_of_unique"] or 0))
    return [{"label": DQ_LABEL[r["metric"]], "value": f"{(r['pct_of_unique'] or 0):.2f}% ({(r['value'] or 0):,.0f})",
             "w": f"{min(r['pct_of_unique'] or 0, 100):.2f}%",
             "cls": "bar-row " + ("pm10" if r["metric"] == "silver_valid" else "muted"),
             "title": f"{DQ_LABEL[r['metric']]}: {(r['value'] or 0):,.0f} readings"} for r in data]


def bars(rows: rx.Var, caption: str | rx.Var = "") -> rx.Component:
    """Horizontal bars: label, track with the value as its length, the value as text."""
    return rx.el.figure(
        rx.el.ul(rx.foreach(rows, lambda r: rx.el.li(
            rx.el.span(r["label"], class_name="bar-label"),
            rx.el.span(rx.el.span(class_name="bar-fill-h", style={"width": r["w"]}), class_name="bar-track"),
            rx.el.span(r["value"], class_name="bar-value"),
            class_name=r["cls"], title=r["title"])), class_name="bars"),
        *([rx.el.figcaption(caption, class_name="caption left")] if caption else []),
        class_name="chart")


# --------------------------------------------------------------------------------------------- calibration (SVG)
# Plot area inside the viewBox: x 46..336, y 14..254 (0–100% on both axes).
X0, X1, Y0, Y1 = 46.0, 336.0, 254.0, 14.0
VIEWBOX = "0 0 352 292"


def _x(p: float) -> float:
    return X0 + (X1 - X0) * p


def _y(p: float) -> float:
    return Y0 + (Y1 - Y0) * p


def calibration_lines(rows: list[dict], pollutants: list[str]) -> list[dict]:
    """One line per pollutant through its bins (operational model, test year) and its direct label at the end."""
    ends = {}
    for p in pollutants:
        data = sorted((r for r in rows if r["pollutant"] == p and r["mean_probability"] is not None
                       and r["observed_frequency"] is not None), key=lambda r: r["bin"])
        if data:
            ends[p] = data
    # Two lines end close together: the label of the line that ends higher goes above its end, the other one below,
    # so a label never sits next to the other series' marker.
    top = min(ends, key=lambda p: _y(ends[p][-1]["observed_frequency"])) if len(ends) > 1 else None
    out = []
    for p, data in ends.items():
        last = data[-1]
        above = p == top if top else p != "PM10"
        out.append({"d": "M" + " L".join(f"{_x(r['mean_probability']):.1f},{_y(r['observed_frequency']):.1f}"
                                         for r in data),
                    "cls": "cal-line " + SERIES[p], "label": POLLUTANT_LABEL[p],
                    "lx": f"{min(_x(last['mean_probability']) + 8, X1 - 4):.1f}",
                    "ly": f"{_y(last['observed_frequency']) + (-8 if above else 14):.1f}"})
    return out


def legend(pollutants: list[str]) -> rx.Component:
    """Key of a chart with two or more series: marker shape (circle / square) and name, text in ink colours."""
    return rx.el.ul(*[rx.el.li(rx.el.span(class_name="swatch " + SERIES[p], aria_hidden="true"), POLLUTANT_LABEL[p])
                      for p in pollutants], class_name="legend", aria_label="Legend")


def calibration_dots(rows: list[dict], pollutants: list[str], min_days: int) -> list[dict]:
    """Every bin as a marker (PM10 circle, PM2.5 square); bins with fewer than ``min_days`` days are faded."""
    out = []
    for p in pollutants:
        for r in sorted((r for r in rows if r["pollutant"] == p and r["mean_probability"] is not None
                         and r["observed_frequency"] is not None), key=lambda r: r["bin"]):
            x, y = _x(r["mean_probability"]), _y(r["observed_frequency"])
            judged = (r["n_days"] or 0) >= min_days
            out.append({"x": f"{x:.1f}", "y": f"{y:.1f}", "sq": f"{x - 4.5:.1f}", "sqy": f"{y - 4.5:.1f}",
                        "shape": "square" if p == "PM2.5" else "circle",
                        "cls": "cal-dot " + SERIES[p] + ("" if judged else " faded"),
                        "title": f"{POLLUTANT_LABEL[p]}, forecasts {pct(r['bin_from'])}–{pct(r['bin_to'])}: mean "
                                 f"forecast {pct(r['mean_probability'])}, exceeded on {pct(r['observed_frequency'])} "
                                 f"of days, {r['n_days']} days" + ("" if judged else " (too few to judge)")})
    return out


def calibration_marker(probability: float | None) -> dict:
    """The vertical line at tomorrow's forecast (City page)."""
    if probability is None:
        return {"x": "", "label": ""}
    x = _x(probability)
    return {"x": f"{x:.1f}", "label": f"tomorrow: {pct(probability)}", "lx": f"{min(max(x, X0 + 30), X1 - 30):.1f}"}


def calibration(lines: rx.Var, dots: rx.Var, marker: rx.Var | None = None, label: str = "") -> rx.Component:
    ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    grid = [rx.el.line(x1=f"{_x(t):.1f}", x2=f"{_x(t):.1f}", y1=f"{Y0}", y2=f"{Y1}", class_name="grid-line")
            for t in ticks] + [rx.el.line(x1=f"{X0}", x2=f"{X1}", y1=f"{_y(t):.1f}", y2=f"{_y(t):.1f}",
                                          class_name="grid-line") for t in ticks]
    tick_labels = ([rx.el.text(pct(t), x=f"{_x(t):.1f}", y=f"{Y0 + 16}", text_anchor="middle", class_name="tick")
                    for t in ticks] + [rx.el.text(pct(t), x=f"{X0 - 6}", y=f"{_y(t) + 4:.1f}", text_anchor="end",
                                                  class_name="tick") for t in ticks])
    parts = [*grid, rx.el.line(x1=f"{X0}", y1=f"{Y0}", x2=f"{X1}", y2=f"{Y1}", class_name="diagonal"), *tick_labels,
             rx.el.text("forecast probability", x=f"{(X0 + X1) / 2}", y="288", text_anchor="middle", class_name="axis-title"),
             # rotated with custom_attrs: Reflex drops a plain ``transform`` prop on <text>
             rx.el.text("observed exceedance frequency", x=f"{-(Y0 + Y1) / 2:.0f}", y="12", text_anchor="middle",
                        class_name="axis-title", custom_attrs={"transform": "rotate(-90)"})]
    if marker is not None:
        parts.append(rx.cond(marker["x"] != "", rx.el.g(
            rx.el.line(x1=marker["x"], x2=marker["x"], y1=f"{Y0}", y2=f"{Y1}", class_name="today-line"),
            rx.el.text(marker["label"], x=marker["lx"], y=f"{Y1 - 2}", text_anchor="middle", class_name="today-label")),
            rx.fragment()))
    parts += [rx.foreach(lines, lambda s: rx.el.path(d=s["d"], class_name=s["cls"])),
              rx.foreach(dots, lambda d: rx.el.g(
                  rx.el.title(d["title"]),
                  rx.cond(d["shape"] == "square",
                          rx.el.rect(x=d["sq"], y=d["sqy"], width="9", height="9", rx="1.5", class_name=d["cls"]),
                          rx.el.circle(cx=d["x"], cy=d["y"], r="5", class_name=d["cls"])))),
              rx.foreach(lines, lambda s: rx.el.text(s["label"], x=s["lx"], y=s["ly"], class_name="series-label"))]
    return rx.el.svg(*parts, view_box=VIEWBOX, class_name="calibration", role="img",
                     aria_label=label or "Calibration chart: forecast probability against observed frequency")
