"""Our own icons for domain ideas Lucide has no icon for, drawn on the Lucide grid: 24 × 24, 2 px stroke, round caps
and joins, no fill – so they sit next to ``rx.icon`` without looking borrowed. Decorative unless a label is given."""

from __future__ import annotations

import reflex as rx


def _svg(*shapes: rx.Component, size: int = 20, label: str | None = None, class_name: str = "") -> rx.Component:
    a11y = {"role": "img", "aria_label": label} if label else {"aria_hidden": "true"}
    return rx.el.svg(*([rx.el.title(label)] if label else []), *shapes, view_box="0 0 24 24", width=str(size),
                     height=str(size), fill="none", stroke="currentColor", stroke_width="2", stroke_linecap="round",
                     stroke_linejoin="round", class_name=f"icon {class_name}".strip(), **a11y)


def mark(size: int = 22, label: str | None = None) -> rx.Component:
    """The smogcast mark: a city under layers of haze."""
    return _svg(rx.el.path(d="M4 21V13h4v8M8 21V9h5v12M13 21v-6h4v6M17 21V12h3v9"),
                rx.el.path(d="M2 5.5h8M13 5.5h9M5 2.5h12"), size=size, label=label, class_name="mark")


def particle(pollutant: str, size: int = 16) -> rx.Component:
    """PM10: one coarse particle with fine ones beside it. PM2.5: only fine particles."""
    if pollutant == "PM10":
        shapes = (rx.el.circle(cx="9", cy="12", r="5"), rx.el.circle(cx="18", cy="7", r="1.5"),
                  rx.el.circle(cx="18.5", cy="16.5", r="1.5"))
    else:
        shapes = (rx.el.circle(cx="7", cy="8", r="2"), rx.el.circle(cx="16", cy="7", r="2"),
                  rx.el.circle(cx="11.5", cy="15.5", r="2"), rx.el.circle(cx="19", cy="16", r="1.2"))
    return _svg(*shapes, size=size)


def station(size: int = 16) -> rx.Component:
    """A measuring station: a cabin on the ground with a mast and an air inlet (funnel) on top."""
    return _svg(rx.el.path(d="M3 21h18M6 21v-7h12v7M12 14V7M9 3h6l-1.5 4h-3z"), size=size)
