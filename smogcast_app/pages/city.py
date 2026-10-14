"""City – one city in detail: tomorrow's forecast for both pollutants in words
(forecast.describe), the calibration of the model with tomorrow's forecast marked, and how the model did in this
city in the 2025 test year.

Rows, not two independent columns – cards, then descriptions, then charts – level with each other, one caption
under both charts; "–" instead of None, a sentence about FAR, help on the metric names; title first, then
the city picker. Route ``/city/<city_id>``; ``/city`` opens the first city.

On phones the scores are two small tables, one per pollutant, with the scores as rows and the four models as columns
(eight cards of seven numbers would make the page about 1,900 px longer), and the meaning of each score below.
"""

from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlparse

import reflex as rx

from smogcast.app import forecast as fc
from smogcast.app import services
from smogcast.app.config import load_config
from smogcast_app.components import charts
from smogcast_app.components.icons import particle
from smogcast_app.components.shell import guard, layout, page_head, skeleton_rows, sql_panel, state_badge, term
from smogcast_app.pages.tomorrow import long_day

log = logging.getLogger(__name__)
C = load_config()
CITIES, ORDER = C["cities"], list(C["run"]["cities"])
WARNING_P = C["thresholds"]["warning_probability"]
MIN_DAYS = C["model"]["acceptance"]["k3_calibration_min_bin_days"]
POLLUTANTS = ("PM10", "PM2.5")

# What the metrics of the city table mean: help on each column name.
METRIC_HELP = {
    "Brier": "Mean squared error of the probability. Lower is better (0 = perfect).",
    "BSS": "Improvement over the calendar average. Above 0 = better than the calendar; 1 = perfect.",
    "POD": "Share of smog days that got a warning the day before. Higher is better.",
    "FAR": "Share of warnings that were false alarms. Lower is better. – = no warnings were issued.",
    "AUC": "How well days are ranked from safest to most dangerous. 1 = perfect, 0.5 = no better than chance.",
}
COLUMNS = [("pollutant", "Pollutant", ""), ("model", "Model", ""), ("days", "Days", ""),
           ("exceed", "Exceedance days", ""), ("brier", "Brier", METRIC_HELP["Brier"]), ("bss", "BSS", METRIC_HELP["BSS"]),
           ("pod", "POD", METRIC_HELP["POD"]), ("far", "FAR", METRIC_HELP["FAR"]), ("auc", "AUC", METRIC_HELP["AUC"])]


def fmt(x, digits: int = 3, percent: bool = False) -> str:
    if x is None:
        return fc.DASH
    return fc.pct(x) if percent else f"{x:.{digits}f}"


def metric_row(x: dict) -> dict[str, str]:
    return {"pollutant": x["pollutant"], "model": charts.MODEL_LABEL.get(x["model"], x["model"]),
            "days": f"{x['n_days']:,}" if x["n_days"] is not None else fc.DASH,
            "exceed": f"{x['n_exceed']:,}" if x["n_exceed"] is not None else fc.DASH,
            "brier": fmt(x["brier"]), "bss": fmt(x["bss"], 2), "pod": fmt(x["pod"], percent=True),
            "far": fmt(x["far"], percent=True), "auc": fmt(x["auc"], 2)}


# Phone tables: the models as columns, under short names explained in the caption; the model in use first.
MODEL_ORDER = ("gbt", "logistic", "climatology", "persistence")
MODEL_SHORT = {"gbt": "GBT", "logistic": "Logistic", "climatology": "Calendar", "persistence": "Like today"}


def by_score(metrics: list[dict], pollutant: str) -> tuple[list[str], list[list[str]]]:
    """(model names, one row per score: [score name, value of each model]) for one pollutant."""
    rows = {x["model"]: metric_row(x) for x in metrics if x["pollutant"] == pollutant}
    models = sorted(rows, key=lambda m: (MODEL_ORDER.index(m) if m in MODEL_ORDER else len(MODEL_ORDER), m))
    return ([MODEL_SHORT.get(m, m) for m in models],
            [[label, *[rows[m][key] for m in models]] for key, label, _ in COLUMNS[2:]])


def city_from(url: str) -> str:
    parts = [p for p in urlparse(url).path.split("/") if p]
    city = parts[1] if len(parts) > 1 else ORDER[0]
    return city if city in CITIES else ORDER[0]


class CityState(rx.State):
    city_id: str = ORDER[0]
    title: str = CITIES[ORDER[0]]["name"]
    target: str = ""
    panel: dict[str, dict[str, str]] = {}
    lines: dict[str, list[dict[str, str]]] = {}
    dots: dict[str, list[dict[str, str]]] = {}
    marker: dict[str, dict[str, str]] = {}
    has_charts: bool = False
    metrics: list[dict[str, str]] = []
    phone_heads: dict[str, list[str]] = {}
    phone_rows: dict[str, list[list[str]]] = {}
    sql_forecast: str = ""
    sql_reliability: str = ""
    sql_metrics: str = ""
    source: str = ""
    loading: bool = True
    error: str = ""

    @rx.event
    def go(self, city_id: str):
        return rx.redirect(f"/city/{city_id}")

    @rx.event(background=True)
    async def load(self, refresh: bool = False):
        async with self:
            self.loading, self.error = True, ""
            city_id = city_from(str(self.router.url))
            self.city_id, self.title = city_id, CITIES[city_id]["name"]
        try:
            data = await asyncio.to_thread(services.gold_data, refresh)
        except Exception:
            log.exception("city data could not be loaded")
            async with self:
                self.loading, self.error = False, "The forecast could not be loaded. Try again in a minute."
            return
        if data.get("forecast") is None:
            async with self:
                self.loading, self.error = False, "There is no forecast yet."
            return
        rows = data["forecast"].rows
        rel = data["reliability"].rows if data.get("reliability") else []
        acc = data["acceptance"].rows if data.get("acceptance") else []
        by_p = {p: next((x for x in rows if x["city_id"] == city_id and x["pollutant"] == p), None) for p in POLLUTANTS}
        panel, lines, dots, marker = {}, {}, {}, {}
        for p, r in by_p.items():
            prob = None if r is None else r["probability"]
            state = "none" if prob is None else ("warning" if r["warned"] else "clear")
            panel[p] = {"has": "" if r is None else "1", "pct": fc.pct(prob), "state": state,
                        "w": f"{0 if prob is None else prob * 100:.1f}%",
                        "text": "" if r is None else fc.describe(r, rel, acc, WARNING_P)}
            lines[p], dots[p] = charts.calibration_lines(rel, [p]), charts.calibration_dots(rel, [p], MIN_DAYS)
            marker[p] = charts.calibration_marker(prob)
        raw = [x for x in (data["metrics"].rows if data.get("metrics") else []) if x["city_id"] == city_id]
        phone = {p: by_score(raw, p) for p in POLLUTANTS}
        async with self:
            self.target = long_day(rows[0]["target_day"]) if rows else ""
            self.panel, self.lines, self.dots, self.marker = panel, lines, dots, marker
            self.has_charts = bool(rel) and any(v["has"] and v["pct"] != fc.DASH for v in panel.values())
            self.metrics = [metric_row(x) for x in raw]
            self.phone_heads = {p: heads for p, (heads, _) in phone.items()}
            self.phone_rows = {p: body for p, (_, body) in phone.items()}
            self.sql_forecast = data["forecast"].sql
            self.sql_reliability = data["reliability"].sql if data.get("reliability") else ""
            self.sql_metrics = data["metrics"].sql if data.get("metrics") else ""
            self.source = data["forecast"].source
            self.loading = False


def picker() -> rx.Component:
    return rx.el.label(
        rx.el.span("City", class_name="picker-label"),
        rx.el.select(*[rx.el.option(CITIES[c]["name"], value=c) for c in ORDER], value=CityState.city_id,
                     on_change=CityState.go, class_name="select", aria_label="City"),
        class_name="picker")


def forecast_card(p: str) -> rx.Component:
    v = CityState.panel[p]
    return rx.el.article(
        rx.el.header(rx.el.h2(particle(p, 18), p, class_name="pair-title"), rx.cond(v["has"] == "1",
                     state_badge(v["state"]), rx.fragment()), class_name="card-head"),
        rx.cond(v["has"] == "1", rx.fragment(
            rx.el.p(v["pct"], class_name="big-value"),
            rx.el.div(rx.el.div(class_name="bar-fill", style={"width": v["w"]}), rx.el.span(class_name="bar-mark"),
                      class_name="bar bar-" + v["state"], aria_hidden="true"),
            rx.el.p(f"chance of exceeding the daily limit tomorrow; a warning from {fc.pct(WARNING_P)}",
                    class_name="card-caption")),
            rx.el.p("No forecast row.", class_name="muted-text")),
        class_name="card pair-card " + p.replace(".", ""))


def description(p: str) -> rx.Component:
    return rx.el.div(rx.markdown(CityState.panel[p]["text"], class_name="prose"), class_name="pair-text " + p.replace(".", ""))


def chart(p: str) -> rx.Component:
    return rx.el.figure(charts.calibration(CityState.lines[p], CityState.dots[p], CityState.marker[p],
                                           f"Calibration of the {p} model in 2025"),
                        class_name="pair-chart " + p.replace(".", ""))


def heading_cell(label: str, hint: str) -> rx.Component:
    """A column name; with a hint, its explanation on hover and on keyboard focus."""
    return term(label, hint) if hint else rx.el.span(label)


def metrics_table() -> rx.Component:
    return rx.el.div(rx.el.table(
        rx.el.thead(rx.el.tr(*[rx.el.th(heading_cell(label, hint), class_name="" if i < 2 else "num")
                               for i, (_, label, hint) in enumerate(COLUMNS)])),
        rx.el.tbody(rx.foreach(CityState.metrics, lambda r: rx.el.tr(
            *[rx.el.td(r[k], class_name="" if i < 2 else "num") for i, (k, _, _) in enumerate(COLUMNS)]))),
        class_name="table"), class_name="table-wrap desktop-only")


def metrics_phone() -> rx.Component:
    """Phone: per pollutant a table of scores × models (fits 390 px without scrolling sideways), then what each score
    means – a touch screen has no hover."""
    tables = [rx.el.table(
        rx.el.caption(p),
        rx.el.thead(rx.el.tr(rx.el.th("Score"), rx.foreach(CityState.phone_heads[p], lambda h: rx.el.th(h, class_name="num")))),
        rx.el.tbody(rx.foreach(CityState.phone_rows[p], lambda row: rx.el.tr(rx.foreach(
            row, lambda cell, i: rx.el.td(cell, class_name=rx.cond(i == 0, "", "num")))))),
        class_name="table compact") for p in POLLUTANTS]
    meanings = rx.el.dl(*[rx.el.div(rx.el.dt(name), rx.el.dd(text)) for name, text in METRIC_HELP.items()],
                        class_name="meanings")
    return rx.el.div(*tables,
                     rx.el.p("Calendar = the calendar average (climatology); Like today = “tomorrow like today” "
                             "(persistence).", class_name="caption left"),
                     meanings, class_name="phone-only metric-cards")


def page() -> rx.Component:
    return layout(
        "/city",
        page_head(CityState.title, rx.cond(CityState.target != "", "Forecast for " + CityState.target +
                                           " and how reliable the model is in this city",
                                           "How reliable the model is in this city"), picker()),
        guard(CityState.error, CityState.loading, CityState.load(True), skeleton_rows(), rx.fragment(
            rx.el.div(*[forecast_card(p) for p in POLLUTANTS], *[description(p) for p in POLLUTANTS],
                      *([chart(p) for p in POLLUTANTS]), class_name="pairs"),
            rx.cond(CityState.has_charts, rx.el.p(
                "Model calibration in 2025: a point on the diagonal = “X%” meant X%. Faded points: fewer than "
                f"{MIN_DAYS} days, too few to judge. The vertical line marks tomorrow's forecast.", class_name="caption"),
                rx.fragment()),
            rx.el.section(
                rx.el.h2("How the model did in this city (2025 test year)", class_name="h2"),
                metrics_table(), metrics_phone(),
                rx.el.p(rx.el.span("Point at a column name, or reach it with the Tab key, to see what it means. ",
                                   class_name="desktop-only"),
                        "– in FAR: that forecast never issued a warning (the calendar average never reaches 50%), so "
                        "there is nothing to count as a false alarm.", class_name="caption left"),
                class_name="section"),
            rx.el.section(sql_panel(CityState.sql_forecast, "SQL behind the forecast"),
                          sql_panel(CityState.sql_reliability, "SQL behind the calibration charts"),
                          sql_panel(CityState.sql_metrics, "SQL behind the scores"), class_name="section tight"),
        )),
        source=CityState.source, refresh=CityState.load(True),
    )
