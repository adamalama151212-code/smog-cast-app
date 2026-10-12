"""The page shell: top bar with navigation (desktop), tab bar (phone), footer with the data sources and the "Refresh
data" button (the Streamlit sidebar), and the pieces every data view shares – page heading, loading and error states,
the SQL panel (A2), the forecast-state badge and the data-source note."""

from __future__ import annotations

import reflex as rx

from smogcast_app import design
from smogcast_app.components.icons import mark

# (route, Lucide icon, label, short label for the phone tab bar) – the Streamlit menu, Info first (author's choice).
NAV = [
    ("/info", "info", "Info", "Info"),
    ("/", "calendar-days", "Tomorrow", "Tomorrow"),
    ("/city", "building-2", "City", "City"),
    ("/model", "chart-no-axes-combined", "Model", "Model"),
    ("/data-quality", "shield-check", "Data quality", "Quality"),
    ("/assistant", "messages-square", "Assistant", "Assistant"),
]
SAMPLE_TAKEN = "9 Oct 2026"   # when the copy in tools/sample was saved


def _link(route: str, icon: str, label: str, active: str, cls: str) -> rx.Component:
    on = route == active
    attrs: dict = {"aria_current": "page"} if on else {}
    return rx.el.a(rx.icon(icon, size=18 if cls == "tab" else 16), rx.el.span(label), href=route,
                   class_name=f"{cls} on" if on else cls, **attrs)


def topbar(active: str) -> rx.Component:
    return rx.el.header(
        rx.el.a(mark(label=None), rx.el.span("smogcast"), href="/", class_name="brand", aria_label="smogcast home"),
        rx.el.nav(*[_link(r, i, label, active, "nav") for r, i, label, _ in NAV], class_name="navs",
                  aria_label="Pages"),
        rx.el.button(rx.color_mode_cond(rx.icon("moon", size=18), rx.icon("sun", size=18)),
                     on_click=rx.toggle_color_mode, class_name="icon-button", aria_label="Switch light or dark theme"),
        class_name="topbar",
    )


def tabbar(active: str) -> rx.Component:
    return rx.el.nav(*[_link(r, i, short, active, "tab") for r, i, _, short in NAV], class_name="tabbar",
                     aria_label="Pages")


def footer(source: rx.Var | None = None, refresh=None) -> rx.Component:
    status = []
    if source is not None:
        status.append(rx.el.span(rx.icon("database", size=14), rx.match(
            source, ("warehouse", "Live data from the result tables"),
            ("sample", f"Saved copy of the result tables, {SAMPLE_TAKEN}"), "Loading the latest forecast…"),
            class_name="status-item"))
    if refresh is not None:
        status.append(rx.el.button(rx.icon("refresh-cw", size=14), "Refresh data", on_click=refresh, type="button",
                                   class_name="link-button"))
    return rx.el.footer(
        rx.el.div(*status, class_name="status") if status else rx.fragment(),
        rx.el.p("Measurements: GIOŚ (Chief Inspectorate of Environmental Protection). Weather forecasts: Open-Meteo. "
                "Every number on these pages is read from the published result tables; each view shows its SQL."),
        class_name="footer",
    )


def layout(active: str, *body: rx.Component, source: rx.Var | None = None, refresh=None) -> rx.Component:
    return rx.el.div(rx.el.a("Skip to content", href="#main", class_name="skip"), topbar(active),
                     rx.el.main(*body, id="main", tab_index=-1, class_name="main"), footer(source, refresh),
                     tabbar(active), class_name="app")


def page_head(title: rx.Var | str, subtitle: rx.Var | str, *extra: rx.Component) -> rx.Component:
    """Title first, then what the page is about (A5: nothing hides under a bar, the title is never cut off)."""
    return rx.el.section(rx.el.h1(title, class_name="h1"), rx.el.p(subtitle, class_name="lead narrow"), *extra,
                         class_name="page-head")


def guard(error: rx.Var, loading: rx.Var, retry, skeleton: rx.Component, body: rx.Component) -> rx.Component:
    """Every data view has three designed states: an error that says what to do, a skeleton while loading, the data."""
    return rx.cond(
        error != "",
        rx.el.div(rx.icon("cloud-off", size=20), rx.el.p(error),
                  rx.el.button(rx.icon("refresh-cw", size=15), "Try again", on_click=retry, class_name="button"),
                  class_name="callout", role="alert"),
        rx.cond(loading, skeleton, body),
    )


def skeleton_rows(n: int = 6, note: str = "Loading the latest forecast…") -> rx.Component:
    """The generic loading state (A10: plain words, no "Spark", no "gold tables")."""
    return rx.el.div(rx.el.p(note, class_name="note"), *[rx.el.div(class_name="sk sk-row") for _ in range(n)],
                     aria_busy="true")


def sql_panel(sql: rx.Var | str, label: str = "SQL behind these numbers") -> rx.Component:
    """A2: the query that produced the numbers, folded away; it can be pasted into the SQL Editor as it is."""
    return rx.el.details(rx.el.summary(rx.icon("code-xml", size=15), label), rx.el.pre(rx.el.code(sql)),
                         class_name="sql")


# Typography of text written outside the app, as the app writes it. The em dash is written as chr(0x2014), never as
# the character: the tools that edit this file turn the escape into the character, and a test forbids it here (A3).
_TYPOGRAPHY = str.maketrans({
    0x2014: 0x2013,   # em dash -> en dash (A3)
    0x2011: "-",      # non-breaking hyphen (gpt-oss writes "missing‑value"; fonts may lack it: "24,hour")
    0x2010: "-",      # hyphen
    0x202F: " ",      # narrow no-break space ("6.69 %")
    0x00A0: " ",      # no-break space
    0x3010: "[",      # 【6】 -> [6]: citations look like the other answers'
    0x3011: "]",
})


def shown(text: str) -> str:
    """Text from outside the app as the user sees it: source passages quote official documents verbatim (the EU
    directives use the em dash as a list mark, A3) and language models write typographic characters of their own –
    all shown in the app's own typography."""
    return text.translate(_TYPOGRAPHY)


def term(label: rx.Var | str, hint: rx.Var | str) -> rx.Component:
    """A term with its explanation (A1, A13): shown on hover and on keyboard focus (CSS, assets/app.css ``.help``) and
    read out by screen readers. Not ``rx.tooltip``: in Reflex 0.10 its Radix trigger opens on hover only, never on
    focus (checked in stage 4), and a ``title`` attribute is mouse-only too."""
    return rx.el.span(label, tab_index=0, class_name="help", data_tip=hint, aria_description=hint)


def state_badge(state: rx.Var | str) -> rx.Component:
    """Forecast state: never colour alone – shape, icon and word together."""
    return rx.match(
        state,
        ("warning", rx.el.span(rx.icon(design.STATE_ICON["warning"], size=14), design.STATE_LABEL["warning"],
                               class_name="badge warning")),
        ("none", rx.el.span(rx.icon(design.STATE_ICON["none"], size=14), design.STATE_LABEL["none"],
                            class_name="badge none")),
        rx.el.span(rx.icon(design.STATE_ICON["clear"], size=14), design.STATE_LABEL["clear"], class_name="badge clear"),
    )


def source_note(source: rx.Var, preview: rx.Var) -> rx.Component:
    """Say plainly when the page shows something other than live data."""
    return rx.cond(
        preview,
        rx.el.p(rx.icon("sparkles", size=14), "Design preview: a day of the 2025 test year, not tomorrow's forecast.",
                class_name="note"),
        rx.cond(source == "sample",
                rx.el.p(rx.icon("database", size=14), f"Saved copy of {SAMPLE_TAKEN}: live data is "
                        "unavailable right now.", class_name="note"),
                rx.fragment()),
    )
