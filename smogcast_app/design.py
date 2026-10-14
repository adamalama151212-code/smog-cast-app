"""Design tokens of the smogcast interface – direction "Atlas".

One source for every colour, size, font and duration. Pages and components never use raw values: they use the CSS
custom properties generated here (``var(--ink)``, ``var(--space-4)`` …) or the Python constants below.
``write_stylesheet()`` writes assets/design.css at start-up, so the CSS can never drift from this file.

Colour has four jobs:
- surfaces and ink (text): quiet, map-like neutrals; text never wears a state colour;
- forecast state: ``clear`` (no warning) and ``warning`` (probability >= 50%) – a validated CVD-safe blue/orange
  pair (dataviz validator, both modes); ``none`` (no forecast) is grey and dashed. A state is never colour alone:
  it always comes with a shape (circle / diamond / dashed ring), an icon and a word;
- magnitude: the probability itself is shown by position and size (halo area on the map, bar length) and by the
  number – never by a colour ramp (a 10-step ramp failed the validator, and colour must not re-encode length);
- pollutant series on charts: ``pm10`` aqua and ``pm25`` violet (validated pair, both modes; never the state colours,
  never on the same chart as a state colour) – always with a direct label and a marker shape as well;
- interface accent: ``focus`` for keyboard focus and links.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------------------------- colour
LIGHT = {
    "page": "#f2f1ec", "surface": "#fbfaf6", "raised": "#ffffff", "sunk": "#e9e7df",
    "line": "#d6d3c8", "line-strong": "#b9b4a5",
    "ink": "#1b1f1d", "ink-2": "#4c524e", "muted": "#5d635e",
    "land": "#e4e1d6", "border": "#b9b4a5",
    "clear": "#2a78d6", "warning": "#eb6834", "none": "#898781",
    "warning-ink": "#1a1206", "warning-soft": "rgba(235, 104, 52, 0.14)", "clear-soft": "rgba(42, 120, 214, 0.12)",
    "focus": "#2a78d6", "code-bg": "#e9e7df", "shadow": "0 1px 2px rgba(27, 31, 29, 0.08)",
    "pm10": "#1baf7a", "pm25": "#4a3aa7", "grid": "#e1e0d9", "axis": "#c3c2b7", "deemphasis": "#c3c2b7",
    # WCAG AA: link text >= 4.5:1 on page/surface/sunk (focus blue is 3.9:1 – fine for a ring, not for
    # text); edges of controls (inputs, examples) >= 3:1 on raised and page
    "link": "#1d62bb", "control": "#8a8577",
}
DARK = {
    "page": "#0f1716", "surface": "#142020", "raised": "#1a2828", "sunk": "#0b1211",
    "line": "#24332f", "line-strong": "#35484b",
    "ink": "#eef2ef", "ink-2": "#b2bdb7", "muted": "#8b9892",
    "land": "#18252a", "border": "#35484b",
    "clear": "#3987e5", "warning": "#d95926", "none": "#898781",
    "warning-ink": "#1a1206", "warning-soft": "rgba(217, 89, 38, 0.20)", "clear-soft": "rgba(57, 135, 229, 0.16)",
    "focus": "#6da7ec", "code-bg": "#0b1211", "shadow": "0 1px 2px rgba(0, 0, 0, 0.4)",
    "pm10": "#199e70", "pm25": "#9085e9", "grid": "#24332f", "axis": "#35484b", "deemphasis": "#4a5a57",
    "link": "#6da7ec", "control": "#62797c",
}

# --------------------------------------------------------------------------------------------- type
FONTS_URL = ("https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600"
             "&family=Source+Sans+3:wght@400;500;600;700&display=swap")
FONT = {
    "display": '"Newsreader", Georgia, serif',          # headlines and the big answer: the "atlas" voice
    "ui": '"Source Sans 3", system-ui, sans-serif',     # everything else; normal spaces around the en dash
    "code": 'ui-monospace, "Cascadia Mono", Consolas, monospace',   # SQL only
}
# size / line height (px). Display steps for headlines, text steps for the interface.
TYPE = {
    "display-1": (48, 52), "display-2": (34, 40), "display-3": (26, 32),
    "lead": (18, 28), "body": (16, 24), "small": (14, 20), "tiny": (13, 18), "micro": (12, 16), "nano": (11, 14),
}
TYPE_PHONE = {"display-1": (34, 40), "display-2": (28, 34), "display-3": (22, 28), "lead": (17, 26)}

# --------------------------------------------------------------------------------------------- space, shape, motion
SPACE = {1: 4, 2: 8, 3: 12, 4: 16, 5: 24, 6: 32, 7: 48, 8: 64}     # 4/8 px grid
RADIUS = {"small": 4, "control": 8, "card": 12, "pill": 999}
MARK = {"thin": 6, "bar": 10}                                      # px: thickness of probability bars / chart bars
WIDTH = {"page": 1280, "text": 640, "thread": 820}                 # thread: the assistant's conversation
TARGET = 40                                                        # px: smallest round button (touch target)
GUTTER = {"desktop": 36, "phone": 16}
PHONE_MAX = 720                                                    # px: below this the phone layout applies
MOTION = {"fast": 160, "base": 240, "slow": 600}                   # ms; all zero under prefers-reduced-motion
EASE = "cubic-bezier(0.2, 0.7, 0.2, 1)"

# --------------------------------------------------------------------------------------------- forecast
WARNING_P = 0.5     # thresholds.warning_probability (conf/thresholds.yaml); shown wherever a probability is
STATE_LABEL = {"warning": "Warning", "clear": "No warning", "none": "No forecast"}
STATE_ICON = {"warning": "triangle-alert", "clear": "check", "none": "circle-dashed"}


def _vars(tokens: dict[str, str]) -> str:
    return " ".join(f"--{k}: {v};" for k, v in tokens.items())


def stylesheet() -> str:
    """CSS custom properties for both themes (light = base, dark under the ``.dark`` class Reflex sets on <html>)."""
    fixed = {**{f"space-{k}": f"{v}px" for k, v in SPACE.items()},
             **{f"radius-{k}": f"{v}px" for k, v in RADIUS.items()},
             **{f"mark-{k}": f"{v}px" for k, v in MARK.items()},
             **{f"font-{k}": v for k, v in FONT.items()},
             **{f"fs-{k}": f"{s}px" for k, (s, _) in TYPE.items()},
             **{f"lh-{k}": f"{h}px" for k, (_, h) in TYPE.items()},
             **{f"{k}-width": f"{v}px" for k, v in WIDTH.items()}, "target": f"{TARGET}px",
             "gutter": f"{GUTTER['desktop']}px",
             **{f"motion-{k}": f"{v}ms" for k, v in MOTION.items()}, "ease": EASE}
    phone = {**{f"fs-{k}": f"{s}px" for k, (s, _) in TYPE_PHONE.items()},
             **{f"lh-{k}": f"{h}px" for k, (_, h) in TYPE_PHONE.items()}, "gutter": f"{GUTTER['phone']}px"}
    still = {f"motion-{k}": "0ms" for k in MOTION}
    return (
        "/* Generated by smogcast_app/design.py – do not edit by hand. */\n"
        f":root {{ {_vars(fixed)} {_vars(LIGHT)} color-scheme: light; }}\n"
        f".dark {{ {_vars(DARK)} color-scheme: dark; }}\n"
        f"@media (max-width: {PHONE_MAX}px) {{ :root {{ {_vars(phone)} }} }}\n"
        f"@media (prefers-reduced-motion: reduce) {{ :root {{ {_vars(still)} }} }}\n"
    )


def write_stylesheet(path: Path = Path(__file__).resolve().parent.parent / "assets" / "design.css") -> str:
    """Write assets/design.css (only when it changed, so the dev server does not reload for nothing)."""
    css = stylesheet()
    if not path.exists() or path.read_text(encoding="utf-8") != css:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(css, encoding="utf-8")
    return "/" + path.name
