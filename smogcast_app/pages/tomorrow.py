"""Tomorrow – the start page: the answer in one sentence, the map of Poland with the ten cities, the ranked list, one
card per city, the model's verdict, the data table and the SQL.

Wording of runs, reasons and verdicts comes from smogcast.app.forecast, so the page and the assistant say the same.

``/tomorrow/preview`` shows a day of the 2025 test year with warnings (gold.backtest_predictions of the operational
model), to see what a smog day looks like in October; the page says so.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import math

import reflex as rx
from reflex.components.el.elements.inline import A

from smogcast.app import forecast as fc
from smogcast.app import services
from smogcast.app.config import load_config
from smogcast_app.components.icons import particle, station
from smogcast_app.components.shell import guard, layout, source_note, sql_panel, state_badge
from smogcast_app.poland_outline import CITY_XY, PATH, VIEWBOX

log = logging.getLogger(__name__)
CFG = load_config()
WARNING_P = CFG["thresholds"]["warning_probability"]
LIMITS = CFG["thresholds"]["daily_limit_ug_m3"]
HALO = 70                    # map units: radius of the halo at 100%
PREVIEW_DAY = "2025-03-19"   # 7 of 20 forecasts with a warning, values from 4% to 97%

PREVIEW_SQL = f"""SELECT p.city_id, n.city_name, p.pollutant, p.issue_day, p.target_day, p.probability, p.warned,
       'ok' AS status, CAST(NULL AS STRING) AS unusable_reason, m.limit_ug_m3,
       m.max_station_mean_ug_m3 AS pm_d1_max_ug_m3, CAST(NULL AS DOUBLE) AS pm_d_morning_max_ug_m3,
       CAST(NULL AS DOUBLE) AS t_mean_c, CAST(NULL AS DOUBLE) AS wind_mean_ms, CAST(NULL AS DOUBLE) AS calm_hours,
       CAST(NULL AS DOUBLE) AS precip_sum_mm, CAST(NULL AS STRING) AS run_kind, CAST(NULL AS TIMESTAMP) AS issued_at
FROM gold.backtest_predictions p
JOIN gold.model_selection s ON s.pollutant = p.pollutant AND s.model = p.model AND s.operational
JOIN (SELECT DISTINCT city_id, city_name FROM gold.forecast_tomorrow) n ON n.city_id = p.city_id
LEFT JOIN gold.measured_daily m ON m.city_id = p.city_id AND m.pollutant = p.pollutant
     AND m.day_cet = date_sub(p.issue_day, 1)
WHERE p.split = 'test' AND p.target_day = DATE '{PREVIEW_DAY}'
ORDER BY p.pollutant, p.probability DESC"""

PLACE_HALO = 12

# --------------------------------------------------------------------------------------------- formatting
def as_date(d) -> dt.date:
    return d if isinstance(d, dt.date) else dt.date.fromisoformat(str(d))


def long_day(d) -> str:
    d = as_date(d)
    return f"{d:%A} {d.day} {d:%B}" + ("" if d.year == dt.date.today().year else f" {d.year}")


def run_text(row: dict) -> str:
    """The real kind of run and its CET time of issue, in the words of forecast.RUN_LABEL."""
    when = fc.issued_cet(row)
    if when is None:
        return "The model's forecast for this day in the 2025 test year" if row.get("run_kind") is None \
            else f"Forecast issued {long_day(row['issue_day'])}"
    kind = fc.RUN_LABEL.get(row.get("run_kind"), "forecast")
    return f"{kind[:1].upper()}{kind[1:]}, issued {long_day(row['issue_day'])} at {when}"


def on_map(lat: float, lon: float) -> tuple[float, float]:
    """A point in the map's units: the outline's equirectangular projection, read off Warszawa and Kraków."""
    (xw, yw), (xk, yk) = CITY_XY["warszawa"], CITY_XY["krakow"]
    w, k = CFG["cities"]["warszawa"], CFG["cities"]["krakow"]
    return (xw + (lon - w["lon"]) * (xk - xw) / (k["lon"] - w["lon"]),
            yw + (lat - w["lat"]) * (yk - yw) / (k["lat"] - w["lat"]))



PLACES = (
    ("Zbąszynek", on_map(52.2436, 15.8139), "https://www.youtube.com/watch?v=UKf4mAN-7ek"),
    ("Dawidy Bankowe", on_map(52.1232, 20.9795), "https://youtu.be/RQsuVdjxNOc?t=98"),
)


def halo(p: float | None) -> float:
    """Halo radius in map units: the AREA grows with the probability (radius with its square root), from zero –
    a 2% city gets almost no halo, so a small chance never looks like a big one."""
    return HALO * math.sqrt(p or 0)


def display(r: dict, rank: int) -> dict[str, str]:
    p = r["probability"]
    state = "none" if p is None else ("warning" if r["warned"] else "clear")
    x, y = CITY_XY.get(r["city_id"], (0, 0))
    east = x > 780                                      # labels of the easternmost cities go to the left of the pin
    limit = r["limit_ug_m3"] or LIMITS.get(r["pollutant"], 0)
    reason = r.get("unusable_reason")
    return {
        "id": r["city_id"], "city": r["city_name"], "pct": fc.pct(p), "state": state, "rank": str(rank),
        "w": f"{0 if p is None else p * 100:.1f}%",
        "x": f"{x}", "y": f"{y}", "r": f"{halo(p):.1f}",
        "lx": f"{x - 18 if east else x + 18}", "ly": f"{y + 7}", "anchor": "end" if east else "start",
        "label": f"{r['city_name']} {fc.pct(p)}",
        "diamond": f"M{x},{y - 12} L{x + 12},{y} L{x},{y + 12} L{x - 12},{y} Z",
        "yday": fc.num(r["pm_d1_max_ug_m3"], 0, "µg/m³"),
        "yday_w": f"{min((r['pm_d1_max_ug_m3'] or 0) / (2 * limit), 1) * 100 if limit else 0:.0f}%",
        "morning": fc.num(r["pm_d_morning_max_ug_m3"], 0, "µg/m³"),
        "temp": fc.num(r["t_mean_c"], 1, "°C"), "wind": fc.num(r["wind_mean_ms"], 1, "m/s"),
        "calm": fc.num(r["calm_hours"], 0, "h"), "rain": fc.num(r["precip_sum_mm"], 1, "mm"),
        "reason": "" if p is not None else fc.REASON.get(reason, reason or "no data"),
    }


def headline(rows: list[dict], pollutant: str) -> list[str]:
    warned = [r for r in rows if r["warned"]]
    ok = [r for r in rows if r["probability"] is not None]
    top = max(ok, key=lambda r: r["probability"]) if ok else None
    if not rows:
        return ["No forecast yet.", ""]
    first = (f"No {pollutant} warnings for tomorrow." if not warned else
             f"{len(warned)} of {len(rows)} cities get a {pollutant} warning for tomorrow.")
    second = (f"The highest chance of exceeding the daily limit is {fc.pct(top['probability'])}, in "
              f"{top['city_name']}." if top else "")
    return [first, second]


def verdicts(acceptance: list[dict]) -> list[dict[str, str]]:
    """The model's verdict on the 2025 test year, in the words of forecast.CRITERIA."""
    out = []
    for p in ("PM10", "PM2.5"):
        passed, failed = fc.verdict(acceptance, p)
        if passed is not None:
            out.append({"pollutant": p, "ok": "1" if passed else "",
                        "text": "passes all five tests" if passed else "; ".join(fc.failed_reasons(failed))})
    return out


def _load(preview: bool, refresh: bool):
    if preview:
        return services.gold_reader(CFG, max_rows=1000).query(PREVIEW_SQL), services.gold_data()
    data = services.gold_data(refresh)
    return data["forecast"], data


# --------------------------------------------------------------------------------------------- state
class TomorrowState(rx.State):
    pollutant: str = "PM10"
    ranked: dict[str, list[dict[str, str]]] = {}    # per pollutant, highest chance first
    mapped: dict[str, list[dict[str, str]]] = {}    # per pollutant, fixed city order (halos change size, not place)
    heads: dict[str, list[str]] = {}
    verdict: list[dict[str, str]] = []
    day: str = ""
    run: str = ""
    sql: str = ""
    source: str = ""
    preview: bool = False
    loading: bool = True
    error: str = ""

    @rx.var
    def rows(self) -> list[dict[str, str]]:
        return self.ranked.get(self.pollutant, [])

    @rx.var
    def pins(self) -> list[dict[str, str]]:
        return self.mapped.get(self.pollutant, [])

    @rx.var
    def head(self) -> list[str]:
        return self.heads.get(self.pollutant, ["", ""])

    @rx.var
    def limit(self) -> str:
        return f"{LIMITS.get(self.pollutant, 0)} µg/m³"

    @rx.event
    def pick(self, pollutant: str):
        self.pollutant = pollutant

    @rx.event(background=True)
    async def load(self, refresh: bool = False):
        async with self:
            self.loading, self.error = True, ""
            self.preview = "/preview" in str(self.router.url)
            preview = self.preview
        try:
            # The connector blocks: a thread keeps the websocket serving other events (a sleeping warehouse needs
            # about 17 s to wake up).
            forecast, data = await asyncio.to_thread(_load, preview, refresh)
        except Exception:
            log.exception("forecast could not be loaded")
            async with self:
                self.loading = False
                self.error = "The forecast could not be loaded. Try again in a minute."
            return
        if forecast is None:
            async with self:
                self.loading, self.error = False, "There is no forecast yet."
            return
        rows = forecast.rows
        by_pol = {p: [r for r in rows if r["pollutant"] == p] for p in ("PM10", "PM2.5")}
        async with self:
            self.ranked = {p: [display(r, i) for i, r in enumerate(rs)]
                           for p, rs in ((p, sorted(rs, key=lambda r: -(r["probability"] or -1)))
                                         for p, rs in by_pol.items())}
            self.mapped = {p: sorted(rs, key=lambda r: r["id"]) for p, rs in self.ranked.items()}
            self.heads = {p: headline(rs, p) for p, rs in by_pol.items()}
            self.verdict = verdicts(data["acceptance"].rows if data.get("acceptance") else [])
            self.day = long_day(rows[0]["target_day"]) if rows else ""
            self.run = run_text(rows[0]) if rows else ""
            self.sql, self.source = forecast.sql, forecast.source
            self.loading = False


# --------------------------------------------------------------------------------------------- pieces
def pollutant_switch() -> rx.Component:
    return rx.el.div(
        *[rx.el.button(particle(p), p, on_click=TomorrowState.pick(p), type="button",
                       aria_pressed=rx.cond(TomorrowState.pollutant == p, "true", "false"),
                       class_name=rx.cond(TomorrowState.pollutant == p, "seg on", "seg")) for p in ("PM10", "PM2.5")],
        class_name="segs", role="group", aria_label="Pollutant")


def shape(state: rx.Var, cls: str) -> rx.Component:
    """The state as a shape for the list: circle = no warning, diamond = warning, dashed ring = no forecast."""
    return rx.el.span(class_name=f"{cls} " + state, aria_hidden="true")


def list_row(r) -> rx.Component:
    return rx.el.li(
        shape(r["state"], "shape"),
        rx.el.span(r["city"], class_name="city"),
        rx.el.span(rx.cond(r["state"] == "warning", rx.icon("triangle-alert", size=14, aria_label="Warning"),
                           rx.fragment()), r["pct"], class_name="value"),
        class_name="rank-row " + r["state"], data_city=r["id"], tab_index=0,
    )


def pin(r) -> rx.Component:
    """One city on the map: halo (area = probability), marker shape (state) and a direct label."""
    return rx.el.g(
        rx.el.title(r["city"], ": ", r["pct"]),
        rx.el.circle(cx=r["x"], cy=r["y"], r=r["r"], class_name="halo " + r["state"], style={"--i": r["rank"]}),
        rx.match(r["state"],
                 ("warning", rx.el.path(d=r["diamond"], class_name="marker warning")),
                 ("none", rx.el.circle(cx=r["x"], cy=r["y"], r="7", class_name="marker none")),
                 rx.el.circle(cx=r["x"], cy=r["y"], r="7", class_name="marker clear")),
        rx.el.text(r["label"], x=r["lx"], y=r["ly"], text_anchor=r["anchor"], class_name="label"),
        class_name="pin " + r["state"], data_city=r["id"],
    )


def place(name: str, xy: tuple[float, float], url: str) -> rx.Component:
    x, y = xy
    hint = f"{name} – opens a video on YouTube in a new tab"
    return A.create(
        rx.el.title(hint),
        rx.el.circle(cx=f"{x:.1f}", cy=f"{y:.1f}", r=f"{PLACE_HALO}", class_name="halo", style={"--i": "10"}),
        rx.el.circle(cx=f"{x:.1f}", cy=f"{y:.1f}", r="7", class_name="place-dot"),
        rx.el.text(name, x=f"{x - 14:.1f}", y=f"{y + 7:.1f}", text_anchor="end", class_name="label"),
        href=url, target="_blank", rel="noopener noreferrer", aria_label=hint, class_name="place",
    )


def shape_key() -> rx.Component:
    """What the marker shapes mean; the warning threshold in words."""
    return rx.el.div(
        rx.el.span(rx.el.span(class_name="shape clear"), f"below {fc.pct(WARNING_P)}: no warning", class_name="key-item"),
        rx.el.span(rx.el.span(class_name="shape warning"), f"{fc.pct(WARNING_P)} or more: warning", class_name="key-item"),
        class_name="map-key", aria_label="Legend")


def the_map() -> rx.Component:
    return rx.el.figure(
        # The places come after the pins, so a city's halo never covers their link. role="group", not "img": an image
        # hides the links inside it from screen readers.
        rx.el.svg(rx.el.path(d=PATH, class_name="land"), rx.foreach(TomorrowState.pins, pin),
                  *[place(*p) for p in PLACES], view_box=VIEWBOX, class_name="map", role="group",
                  aria_label="Map of Poland with tomorrow's chance of exceeding the daily limit in 10 cities"),
        rx.el.figcaption("The bigger the halo, the higher the chance of exceeding the daily limit tomorrow.",
                         class_name="caption"),
        # In autumn every halo is small; the preview shows a real smog day of the test year
        rx.el.p(rx.cond(TomorrowState.preview,
                        rx.el.a("Back to tomorrow's forecast", href="/", class_name="link"),
                        rx.el.a(f"See what a smog day looked like: {long_day(PREVIEW_DAY)}", href="/tomorrow/preview",
                                class_name="link")),
                class_name="caption"),
        class_name="map-figure")


def bar(width: rx.Var, cls: str = "") -> rx.Component:
    """A 0–100% track with the 50% mark (the warning threshold, or the daily limit on a 0–2× limit scale)."""
    return rx.el.div(rx.el.div(class_name="bar-fill", style={"width": width}), rx.el.span(class_name="bar-mark"),
                     class_name=f"bar {cls}".strip(), aria_hidden="true")


def city_card(r) -> rx.Component:
    return rx.el.article(
        rx.el.header(rx.el.h3(rx.el.a(r["city"], href="/city/" + r["id"]), class_name="card-title"),
                     state_badge(r["state"]), class_name="card-head"),
        rx.el.p(r["pct"], class_name="card-value"),
        bar(r["w"], "bar-" + r["state"]),
        rx.el.dl(
            rx.el.div(rx.el.dt(station(14), "Yesterday"), rx.el.dd(r["yday"]), class_name="fact"),
            bar(r["yday_w"], "bar-measure"),
            rx.el.div(rx.el.dt("This morning"), rx.el.dd(r["morning"]), class_name="fact"),
            class_name="facts"),
        rx.cond(r["reason"] != "", rx.el.p(r["reason"], class_name="card-caption"), rx.fragment()),
        class_name="card city-card " + r["state"],
    )


def model_check() -> rx.Component:
    return rx.el.section(
        rx.el.h2("Can the forecast be trusted?", class_name="h2"),
        rx.el.p("The model was tested on 2025, a year it had never seen, with five tests fixed before training – "
                "see them on the ", rx.el.a("Model", href="/model", class_name="link"), " page.",
                class_name="muted-text"),
        rx.el.ul(rx.foreach(TomorrowState.verdict, lambda v: rx.el.li(
            rx.cond(v["ok"] == "1", rx.icon("circle-check", size=18, class_name="ok"),
                    rx.icon("circle-x", size=18, class_name="bad")),
            rx.el.strong(v["pollutant"], ":"), rx.el.span(v["text"]), class_name="verdict")), class_name="verdicts"),
        class_name="section")


def data_table() -> rx.Component:
    heads = ["City", "Chance", "State", "Yesterday, worst station", "This morning", "Temperature", "Wind",
             "Calm wind", "Rain"]
    keys = ["city", "pct", "state", "yday", "morning", "temp", "wind", "calm", "rain"]
    numeric = (1, 3, 4, 5, 6, 7, 8)
    return rx.el.details(
        rx.el.summary(rx.icon("table-2", size=15), "Data table"),
        rx.el.div(rx.el.table(
            rx.el.thead(rx.el.tr(*[rx.el.th(h, class_name="num" if i in numeric else "") for i, h in enumerate(heads)])),
            rx.el.tbody(rx.foreach(TomorrowState.rows, lambda r: rx.el.tr(
                *[rx.el.td(r[k], class_name="num" if i in numeric else "") for i, k in enumerate(keys)]))),
            class_name="table"), class_name="table-wrap"),
        class_name="disclosure desktop-only")


def skeleton() -> rx.Component:
    return rx.el.div(
        rx.el.div(rx.el.div(class_name="sk sk-line short"), rx.el.div(class_name="sk sk-title"),
                  rx.el.div(class_name="sk sk-line"), rx.el.p("Loading the latest forecast…", class_name="note"),
                  rx.el.div(*[rx.el.div(class_name="sk sk-row") for _ in range(6)]), class_name="hero-text"),
        rx.el.div(rx.el.svg(rx.el.path(d=PATH, class_name="land sk-land"), view_box=VIEWBOX, class_name="map",
                            aria_hidden="true"), class_name="map-figure"),
        class_name="hero", aria_busy="true")


def hero() -> rx.Component:
    return rx.el.section(
        rx.el.div(
            rx.el.p(rx.cond(TomorrowState.preview, "", "Tomorrow, "), TomorrowState.day, class_name="eyebrow"),
            rx.el.h1(TomorrowState.head[0], class_name="h1"),
            rx.el.p(TomorrowState.head[1], class_name="lead"),
            rx.el.p(rx.icon("clock", size=14), TomorrowState.run, class_name="note"),
            source_note(TomorrowState.source, TomorrowState.preview),
            pollutant_switch(),
            rx.el.ol(rx.foreach(TomorrowState.rows, list_row), class_name="rank",
                     aria_label="Cities by chance of exceeding the daily limit"),
            shape_key(),
            class_name="hero-text"),
        the_map(),
        class_name="hero")


def page() -> rx.Component:
    return layout(
        "/",
        guard(TomorrowState.error, TomorrowState.loading, TomorrowState.load(True), skeleton(), rx.fragment(
            hero(),
            rx.el.section(
                rx.el.h2("Every city", class_name="h2"),
                rx.el.p("Chance of exceeding the daily limit of ", TomorrowState.limit,
                        f". The mark on the first bar is {fc.pct(WARNING_P)}: from there on a city gets a warning. "
                        "Yesterday: the highest daily mean among the city's stations; the mark on its bar is the limit.",
                        class_name="muted-text"),
                rx.el.div(rx.foreach(TomorrowState.rows, city_card), class_name="cards"),
                class_name="section"),
            model_check(),
            rx.el.section(data_table(), sql_panel(TomorrowState.sql), class_name="section tight"),
        )),
        source=TomorrowState.source, refresh=TomorrowState.load(True),
    )
