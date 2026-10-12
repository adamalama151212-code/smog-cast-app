"""smogcast – will the daily PM10 / PM2.5 limit be exceeded tomorrow, and how sure is that?

App entry: theme, fonts, design tokens, the six pages of the Streamlit app (Info first in the menu, Tomorrow the start
page) and the assistant prepared when the server starts.
"""

from __future__ import annotations

import asyncio
import logging

import reflex as rx

from smogcast.app import services
from smogcast_app import design
from smogcast_app.pages import assistant, city, data_quality, info, model, tomorrow

TITLE = "smogcast – tomorrow's smog forecast"
log = logging.getLogger(__name__)

app = rx.App(stylesheets=[design.FONTS_URL, design.write_stylesheet(), "/app.css"], html_lang="en")
app.add_page(info.page, route="/info", title=f"How it works – {TITLE}", on_load=info.InfoState.load)
app.add_page(tomorrow.page, route="/", title=TITLE, on_load=tomorrow.TomorrowState.load)
app.add_page(tomorrow.page, route="/tomorrow/preview", title=f"Smog day preview – {TITLE}",
             on_load=tomorrow.TomorrowState.load)
app.add_page(city.page, route="/city", title=f"City – {TITLE}", on_load=city.CityState.load)
app.add_page(city.page, route="/city/[city_id]", title=f"City – {TITLE}", on_load=city.CityState.load)
app.add_page(model.page, route="/model", title=f"Model – {TITLE}", on_load=model.ModelState.load)
app.add_page(data_quality.page, route="/data-quality", title=f"Data quality – {TITLE}",
             on_load=data_quality.QualityState.load)
app.add_page(assistant.page, route="/assistant", title=f"Assistant – {TITLE}", on_load=assistant.AssistantState.open)


@app.register_lifespan_task
async def prepare_assistant():
    """Load the assistant and its embedding model when the server starts (about 10 s, in a worker thread)."""
    try:
        await asyncio.to_thread(services.warm_up)
    except Exception:   # the pages work without the assistant; its own page says it is unavailable
        log.exception("the assistant could not be prepared")
