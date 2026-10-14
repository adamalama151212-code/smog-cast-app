"""Info – how the forecast is made, in plain words.

Thresholds and settings come from the configuration and the verdict from gold, so the page cannot drift from what the
pipeline really does. No list of K1–K4 here, the tests are on the Model page. The four steps are equal tiles
in one grid (two by two, then one column, on narrow screens).
"""

from __future__ import annotations

import asyncio
import logging

import reflex as rx

from smogcast.app import forecast as fc
from smogcast.app import services
from smogcast.app.config import load_config
from smogcast.core.schema import CALENDAR_FEATURES, PM_FEATURES, WEATHER_FEATURES
from smogcast_app.components.shell import layout, page_head

log = logging.getLogger(__name__)
C = load_config()
LIMITS, WARNING_P = C["thresholds"]["daily_limit_ug_m3"], C["thresholds"]["warning_probability"]
M = C["model"]
MORNING_END = M["morning_last_hour_cet"] + 1
SPLIT = M["split"]
TRAIN_FROM, TRAIN_TO = SPLIT["train_years"]
VAL_YEAR, TEST_YEAR = SPLIT["validation_year"], SPLIT["test_year"]

QUESTION = (
    f"**Will the daily limit be exceeded tomorrow in a city, and how sure is that?** A day counts as an exceedance "
    f"when the daily mean concentration at **at least one station** of the city is above the limit: "
    f"**PM10 > {LIMITS['PM10']} µg/m³** (the Polish and EU limit value) or **PM2.5 > {LIMITS['PM2.5']} µg/m³** (the EU "
    f"daily limit that applies from 2030; Polish law has no daily PM2.5 limit today). A station's daily mean counts "
    f"only if at least {C['thresholds']['daily_min_valid_hours']} of its 24 hourly readings exist. The answer is a "
    f"**probability from 0 to 100%**, separately for PM10 and PM2.5 in each of the {len(C['run']['cities'])} largest "
    f"cities in Poland.")
STEPS = [
    ("Today's air", f"Every day at {M['official_from_cet']} CET (the official forecast) the model takes the measurements "
                    f"available at that moment: yesterday's daily means and this morning's readings "
                    f"(00:00–{MORNING_END:02d}:00)."),
    ("Tomorrow's weather", "The weather forecast for tomorrow from Open-Meteo: temperature, wind, hours of calm wind, "
                           "rain, humidity, pressure, clouds and sunshine."),
    ("The model", "A gradient-boosted trees (GBT) model, trained on years of past days, combines these inputs with the "
                  "month, the weekend and the city."),
    ("Probability and warning", f"The result is the probability of an exceedance tomorrow. At {fc.pct(WARNING_P)} or "
                                f"more the city gets a warning."),
]
LOOKS_AT = (
    "- **Air pollution so far:** yesterday's daily mean (the worst station and the city average), whether yesterday "
    "was already above the limit, the day before yesterday, and this morning's readings. Smog rarely appears or "
    "disappears overnight, so these are the strongest clues.\n"
    "- **Tomorrow's weather forecast:** cold days mean more home heating; calm wind (below 1.5 m/s) and no rain let "
    "smoke stay over the city; wind and rain clear it away.\n"
    "- **Calendar:** month, weekend, heating season (October–April).\n"
    "- **City:** each city has its own typical level of pollution.\n\n"
    "Nothing from tomorrow itself is used. When the model was tested on past years, it was given the weather forecasts "
    "that really existed one day earlier, not the weather that actually happened, so the test shows how good the "
    "forecast is in real use.")
PROBABILITY = (
    "The model is a set of **decision trees** (gradient-boosted trees, GBT). One tree asks a few yes/no questions about "
    "the inputs, for example *“was yesterday above 40 µg/m³?”*, *“will there be more than 8 hours of calm wind?”*, "
    "*“is it the heating season?”*, and ends with a score. The trees are built one after another, and each new tree "
    "learns to correct the mistakes the previous ones made on past days. The scores of all trees are added up and "
    "turned into a probability between 0% and 100%.\n\n"
    f"The model learned from history: the GIOŚ archive of hourly measurements for {TRAIN_FROM}–{VAL_YEAR}, every day "
    "paired with what happened the next day. It is trained once and saved; every day the same saved model computes the "
    "forecast for tomorrow from fresh data.\n\n"
    f"The **probability is always shown**. The warning (≥ {fc.pct(WARNING_P)}) is only a yes/no summary for those who "
    "need one.")
TRUST = (
    f"The model was tested on **{TEST_YEAR}**, a year it had never seen. It must do better than two simple rules: the "
    "**calendar average** (*how often the limit was exceeded in this city in this month*) and **“tomorrow like "
    "today”** (*repeat the last full day*). Five tests were fixed before training – see them on the *Model* page.")
SOURCES = (
    "- **GIOŚ archive** (Chief Inspectorate of Environmental Protection): hourly PM10 and PM2.5 from all stations, "
    f"{TRAIN_FROM}–{TEST_YEAR}, the history the model learned from and was tested on.\n"
    "- **GIOŚ live API:** today's measurements, fetched every hour and cleaned (duplicates, missing and impossible "
    "values, frozen sensors, spikes) before they reach the forecast, see *Data quality*.\n"
    "- **Open-Meteo:** weather forecasts, archived ones for the history and the current one for tomorrow.\n"
    "- **Assistant:** numbers come only from the result tables (the SQL is always shown), knowledge only from "
    "documents (the sources are always shown).")
GRIDS = "; ".join(f"{m}: {g}" for m, g in M["grids"].items())
TECHNICAL = (
    f"- **Inputs:** {', '.join(f'`{x}`' for x in PM_FEATURES + WEATHER_FEATURES + CALENDAR_FEATURES)}, plus the city "
    "as a one-hot vector. Concentrations enter as log(1 + x); wind direction and month as sine/cosine (359° is next to "
    "1°, December next to January).\n"
    f"- **Models compared:** {', '.join(M['models'])}; parameter grids: {GRIDS}.\n"
    f"- **Choice:** trained on {TRAIN_FROM}–{TRAIN_TO}, the model type and settings with the lowest Brier score on "
    f"{VAL_YEAR} win (rule fixed before training); then refit on {TRAIN_FROM}–{VAL_YEAR} and evaluated once on "
    f"{TEST_YEAR}.\n"
    "- **Brier score** = mean squared error of the probability (0 = perfect). **BSS** = improvement over the calendar "
    "average. **POD** = share of smog days that got a warning. **FAR** = share of warnings that were false alarms. "
    "**AUC** = how well the model ranks days from least to most dangerous.\n"
    "- **Engine:** Spark MLlib; the forecast runs as a step of the live pipeline and writes `gold.forecast_tomorrow`.")


def badges(acceptance: list[dict]) -> list[dict[str, str]]:
    out = []
    for p in ("PM10", "PM2.5"):
        passed, failed = fc.verdict(acceptance, p)
        if passed is not None:
            out.append({"ok": "1" if passed else "",
                        "text": f"{p}: passes all tests" if passed else f"{p}: " + "; ".join(fc.failed_reasons(failed))})
    return out


class InfoState(rx.State):
    verdict: list[dict[str, str]] = []
    source: str = ""

    @rx.event(background=True)
    async def load(self, refresh: bool = False):
        try:
            data = await asyncio.to_thread(services.gold_data, refresh)
        except Exception:
            log.exception("verdict could not be loaded")   # the page is text; only the badges are missing
            return
        async with self:
            acc = data.get("acceptance")
            self.verdict = badges(acc.rows if acc else [])
            self.source = acc.source if acc else ""


def prose(markdown: str) -> rx.Component:
    return rx.markdown(markdown, class_name="prose")


def section(title: str, *body: rx.Component) -> rx.Component:
    return rx.el.section(rx.el.h2(title, class_name="h2"), *body, class_name="section")


def page() -> rx.Component:
    return layout(
        "/info",
        page_head("How smogcast works", "How tomorrow's smog forecast is made, in plain words. The other pages show "
                                        "the numbers; this one explains what they mean."),
        section("The question", prose(QUESTION)),
        section("How tomorrow's forecast is made", rx.el.ol(
            *[rx.el.li(rx.el.span(f"Step {i}", class_name="step-n"), rx.el.h3(title, class_name="step-title"),
                       rx.el.p(body, class_name="step-body"), class_name="card step")
              for i, (title, body) in enumerate(STEPS, 1)], class_name="steps")),
        section("What the model looks at", prose(LOOKS_AT)),
        section("How the model turns this into a probability", prose(PROBABILITY)),
        section("How far can it be trusted?", prose(TRUST), rx.el.div(rx.foreach(InfoState.verdict, lambda v: rx.el.span(
            rx.cond(v["ok"] == "1", rx.icon("circle-check", size=15, class_name="ok"),
                    rx.icon("circle-x", size=15, class_name="bad")), v["text"], class_name="verdict-badge")),
            class_name="badges")),
        section("Where the data comes from", prose(SOURCES)),
        rx.el.details(rx.el.summary(rx.icon("settings-2", size=15), "Technical details"), prose(TECHNICAL),
                      class_name="disclosure section"),
        source=InfoState.source, refresh=InfoState.load(True),
    )
