"""Data quality – what happened to the stream readings (the Streamlit page view_data_quality): how many were valid,
how many were rejected and why, for the live GIOŚ feed and for the fault demonstration."""

from __future__ import annotations

import asyncio
import logging

import reflex as rx

from smogcast.app import services
from smogcast_app.components import charts
from smogcast_app.components.shell import guard, layout, page_head, skeleton_rows, sql_panel

log = logging.getLogger(__name__)
LABELS = {"live": "Live GIOŚ API", "fault_replay": "Fault demo (replay with injected faults)"}


def count(x: float) -> str:
    return f"{x:,.0f}"


def tiles(rows: list[dict], source: str) -> list[dict[str, str]]:
    m = {r["metric"]: r for r in rows if r["source"] == source}

    def v(key: str) -> float:
        return (m[key]["value"] or 0.0) if key in m else 0.0

    quarantined, late = v("quarantined"), v("late_rejected")
    valid_pct = (m.get("silver_valid") or {}).get("pct_of_unique") or 0
    return [
        {"label": "Records received", "value": count(v("records_received")),
         "note": f"of which re-sent: {count(v('resent_duplicates'))}"},
        {"label": "Unique readings", "value": count(v("unique_readings")), "note": "station × hour"},
        {"label": "Valid", "value": f"{valid_pct:.1f}%", "note": f"{count(v('silver_valid'))} readings"},
        {"label": "Rejected", "value": count(quarantined + late), "note": f"quarantine {quarantined:.0f} · late {late:.0f}"},
    ]


class QualityState(rx.State):
    sources: list[list[str]] = []          # [id, label]
    source_id: str = ""
    tiles: dict[str, list[dict[str, str]]] = {}
    bars: dict[str, list[dict[str, str]]] = {}
    sql: str = ""
    data_source: str = ""
    loading: bool = True
    error: str = ""

    @rx.event
    def pick(self, source_id: str):
        self.source_id = source_id

    @rx.event(background=True)
    async def load(self, refresh: bool = False):
        async with self:
            self.loading, self.error = True, ""
        try:
            data = await asyncio.to_thread(services.gold_data, refresh)
        except Exception:
            log.exception("data quality could not be loaded")
            async with self:
                self.loading, self.error = False, "The data-quality summary could not be loaded. Try again in a minute."
            return
        if data.get("dq") is None:
            async with self:
                self.loading, self.error = False, "There is no data-quality summary yet."
            return
        rows = data["dq"].rows
        # The live feed first, as a button and as the default (author, 2026-10-09): the page is about live data.
        ids = sorted({r["source"] for r in rows}, key=lambda s: (s != "live", s))
        async with self:
            self.sources = [[s, LABELS.get(s, s)] for s in ids]
            if self.source_id not in ids:
                self.source_id = ids[0] if ids else ""
            self.tiles = {s: tiles(rows, s) for s in ids}
            self.bars = {s: charts.dq_rows(rows, s) for s in ids}
            self.sql, self.data_source = data["dq"].sql, data["dq"].source
            self.loading = False


def tile(t) -> rx.Component:
    return rx.el.div(rx.el.p(t["label"], class_name="tile-label"), rx.el.p(t["value"], class_name="tile-value"),
                     rx.el.p(t["note"], class_name="tile-note"), class_name="card tile")


def page() -> rx.Component:
    return layout(
        "/data-quality",
        page_head("Live data quality", "What happened to the stream readings: how many were valid, how many were "
                                       "rejected and why."),
        guard(QualityState.error, QualityState.loading, QualityState.load(True), skeleton_rows(), rx.fragment(
            rx.el.div(rx.foreach(QualityState.sources, lambda s: rx.el.button(
                s[1], on_click=QualityState.pick(s[0]), type="button",
                aria_pressed=rx.cond(QualityState.source_id == s[0], "true", "false"),
                class_name=rx.cond(QualityState.source_id == s[0], "seg on", "seg"))), class_name="segs",
                role="group", aria_label="Source"),
            rx.el.div(rx.foreach(QualityState.tiles[QualityState.source_id], tile), class_name="tiles"),
            rx.el.section(rx.el.h2("What happened to the unique readings", class_name="h2"),
                          charts.bars(QualityState.bars[QualityState.source_id],
                                      "Share of unique readings (station × hour); the number of readings in brackets."),
                          class_name="section"),
            rx.cond(QualityState.source_id == "fault_replay", rx.el.p(
                "Demo: real measurements from Kraków (17–21 Jan 2025) with deliberately injected faults. The real smog "
                "episode of 20 January passes without flags: a real signal, not a fault.", class_name="caption left"),
                rx.fragment()),
            rx.el.section(sql_panel(QualityState.sql), class_name="section tight"),
        )),
        source=QualityState.data_source, refresh=QualityState.load(True),
    )
