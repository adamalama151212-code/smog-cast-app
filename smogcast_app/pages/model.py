"""Model – how good the model is (the Streamlit page view_model): the five tests fixed before training, the Brier
score of every model and the calibration, on the 2025 test year.

A12: rows, not two independent columns – headings and verdicts, then the two tables, then the two charts, one caption;
both tables have the same rows ("– (PM10 only)"); the Brier charts have their own scales. A13: each test is a question
in plain words, its code only a small grey label, how it is measured on hover.
"""

from __future__ import annotations

import asyncio
import logging

import reflex as rx

from smogcast.app import forecast as fc
from smogcast.app import services
from smogcast.app.config import load_config
from smogcast_app.components import charts
from smogcast_app.components.icons import particle
from smogcast_app.components.shell import guard, layout, page_head, skeleton_rows, sql_panel, term

log = logging.getLogger(__name__)
C = load_config()
K3 = C["model"]["acceptance"]
MIN_DAYS, MAX_GAP = K3["k3_calibration_min_bin_days"], K3["k3_calibration_max_abs_gap"]
POLLUTANTS = ("PM10", "PM2.5")
HOW = {code: c.how for code, c in fc.CRITERIA.items()}


def test_row(cells: list[str]) -> dict[str, str]:
    code, question, value, required, result = cells
    return {"code": code, "question": question, "value": value, "required": required, "result": result,
            "how": HOW.get(code, ""), "cls": "met" if result.startswith("✓") else
            ("na" if "PM10 only" in result else "not-met")}


class ModelState(rx.State):
    heading: dict[str, str] = {}
    passed: dict[str, str] = {}
    tests: dict[str, list[dict[str, str]]] = {}
    brier: dict[str, list[dict[str, str]]] = {}
    lines: list[dict[str, str]] = []
    dots: list[dict[str, str]] = []
    explain: str = ""
    sql_acceptance: str = ""
    sql_metrics: str = ""
    sql_reliability: str = ""
    source: str = ""
    loading: bool = True
    error: str = ""

    @rx.event(background=True)
    async def load(self, refresh: bool = False):
        async with self:
            self.loading, self.error = True, ""
        try:
            data = await asyncio.to_thread(services.gold_data, refresh)
        except Exception:
            log.exception("model data could not be loaded")
            async with self:
                self.loading, self.error = False, "The results could not be loaded. Try again in a minute."
            return
        if data.get("acceptance") is None or data.get("metrics") is None:
            async with self:
                self.loading, self.error = False, "There are no model results yet."
            return
        acc, metrics = data["acceptance"].rows, data["metrics"].rows
        rel = data["reliability"].rows if data.get("reliability") else []
        operational = {p: next((r["model"] for r in acc if r["pollutant"] == p), None) for p in POLLUTANTS}
        async with self:
            self.heading = {p: f"{p}, {str(operational[p]).upper()} model" for p in POLLUTANTS}
            self.passed = {p: "1" if fc.verdict(acc, p)[0] else "" for p in POLLUTANTS}
            self.tests = {p: [test_row(c) for c in fc.criteria_rows(acc, p)] for p in POLLUTANTS}
            self.brier = {p: charts.brier_rows(metrics, p, operational[p]) for p in POLLUTANTS}
            self.lines = charts.calibration_lines(rel, list(POLLUTANTS))
            self.dots = charts.calibration_dots(rel, list(POLLUTANTS), MIN_DAYS)
            self.explain = fc.calibration_text(rel, MIN_DAYS, MAX_GAP) if rel else ""
            self.sql_acceptance = data["acceptance"].sql
            self.sql_metrics = data["metrics"].sql
            self.sql_reliability = data["reliability"].sql if data.get("reliability") else ""
            self.source = data["acceptance"].source
            self.loading = False


def heading(p: str) -> rx.Component:
    return rx.el.header(
        rx.el.h2(particle(p, 18), ModelState.heading[p], class_name="pair-title"),
        rx.cond(ModelState.passed[p] == "1",
                rx.el.span(rx.icon("circle-check", size=14), "passes all tests", class_name="badge clear"),
                rx.el.span(rx.icon("circle-x", size=14), "does not pass all tests", class_name="badge warning")),
        class_name="pair-head " + p.replace(".", ""))


def tests_table(p: str) -> rx.Component:
    """The tests as rows: the question with its code as a small label (how it is measured on hover), the model's
    value and the requirement right-aligned, the result never wrapped."""
    return rx.el.div(
        rx.el.table(
            rx.el.thead(rx.el.tr(rx.el.th("Test"), rx.el.th("Model", class_name="num"),
                                 rx.el.th("Required", class_name="num"), rx.el.th("Result"))),
            rx.el.tbody(rx.foreach(ModelState.tests[p], lambda t: rx.el.tr(
                rx.el.td(term(t["question"], t["how"]), rx.el.span(t["code"], class_name="code-label")),
                rx.el.td(t["value"], class_name="num"), rx.el.td(t["required"], class_name="num"),
                rx.el.td(t["result"], class_name="nowrap result " + t["cls"])))),
            class_name="table"),
        rx.el.div(rx.foreach(ModelState.tests[p], lambda t: rx.el.article(
            rx.el.p(t["question"], rx.el.span(t["code"], class_name="code-label"), class_name="test-q"),
            rx.el.dl(rx.el.div(rx.el.dt("Model"), rx.el.dd(t["value"]), class_name="fact"),
                     rx.el.div(rx.el.dt("Required"), rx.el.dd(t["required"]), class_name="fact"),
                     class_name="facts"),
            rx.el.p(t["result"], class_name="result " + t["cls"]),
            rx.el.p(t["how"], class_name="card-caption"),
            class_name="card test-card")), class_name="phone-only test-cards"),
        class_name="pair-table " + p.replace(".", ""))


def brier(p: str) -> rx.Component:
    return rx.el.div(charts.bars(ModelState.brier[p]), class_name="pair-chart " + p.replace(".", ""))


def page() -> rx.Component:
    return layout(
        "/model",
        page_head("How good is the model?", "The 2025 test year, never seen by the model; weather from real forecasts "
                                            "issued one day earlier. Tests fixed before training."),
        guard(ModelState.error, ModelState.loading, ModelState.load(True), skeleton_rows(), rx.fragment(
            rx.el.p(rx.el.strong(fc.CRITERIA_INTRO), " Point at a test, or reach it with the Tab key, to see how it "
                    "is measured.", class_name="lead narrow intro"),
            rx.el.div(*[heading(p) for p in POLLUTANTS], *[tests_table(p) for p in POLLUTANTS],
                      *[brier(p) for p in POLLUTANTS], class_name="pairs"),
            rx.el.p("Brier score on the 2025 test year – lower is better. ★ = the model in use. Scales differ – compare "
                    "models within one pollutant, not PM10 with PM2.5.", class_name="caption"),
            rx.el.section(
                rx.el.h2("Calibration: does “70%” mean 70%?", class_name="h2"),
                rx.el.div(rx.el.figure(charts.legend(list(POLLUTANTS)),
                                       charts.calibration(ModelState.lines, ModelState.dots,
                                                          label="Calibration of both models in 2025"),
                                       class_name="cal-figure"),
                          rx.markdown(ModelState.explain, class_name="prose"), class_name="cal-layout"),
                class_name="section"),
            rx.el.div(rx.icon("info", size=18),
                      rx.el.p("PM10 is harder to forecast than PM2.5: PM10 also contains dust from streets, construction "
                              "sites and soil, which the weather forecast does not show, and its exceedances are rarer. "
                              "The PM10 model ranks days well from least to most dangerous and its warnings are usually "
                              "right, but it misses many smog days. Ask the assistant for details."),
                      class_name="callout info section"),
            rx.el.section(sql_panel(ModelState.sql_acceptance, "SQL behind the tests"),
                          sql_panel(ModelState.sql_metrics, "SQL behind the Brier scores"),
                          sql_panel(ModelState.sql_reliability, "SQL behind the calibration chart"),
                          class_name="section tight"),
        )),
        source=ModelState.source, refresh=ModelState.load(True),
    )
