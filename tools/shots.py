"""Smoke test + screenshots: open every page in every view, wait until its data is shown, fail on console errors
or a page wider than the screen.

    docker compose --profile tools run --rm shots python tools/shots.py [page names…]

The same test against the hosted app: set its address in the shell (the local container stays the default).

    SMOGCAST_BASE_URL=https://<address> docker compose --profile tools run --rm shots python tools/shots.py

Dark is the app's default; the light views set the stored colour mode first (react-theme.js keeps the user's choice
only while ``last_compiled_theme`` matches the compiled default).
"""

import os
import sys
import time

from playwright.sync_api import sync_playwright

BASE = os.environ.get("SMOGCAST_BASE_URL", "http://localhost:3000").rstrip("/")
# name -> (path, selector that appears once the page shows its data or its designed error state)
PAGES = {
    "info": ("/info", ".step"),
    "tomorrow": ("/", ".rank-row, .callout"),
    "tomorrow-preview": ("/tomorrow/preview", ".rank-row, .callout"),
    "city": ("/city/krakow", ".pair-card, .callout"),
    "model": ("/model", ".pair-head, .callout"),
    "data-quality": ("/data-quality", ".tile, .callout"),
    "assistant": ("/assistant", ".topic"),
}
# A real conversation on the Assistant page: one question per answer route that works without a language model, and
# a follow-up the language model has to read from the conversation ("Question understood by the language model").
CONVERSATION = ["Will there be smog in Kraków tomorrow?", "And in Wrocław?", "What does calm_hours mean?",
                "What is the PM10 alarm level?"]
VIEWS = [("desktop-dark", 1440, 900, "dark"), ("desktop-light", 1440, 900, "light"),
         ("phone-dark", 390, 844, "dark"), ("phone-light", 390, 844, "light")]
# The dev server re-optimises its dependencies after a restart and answers the first request with this; not an app error.
DEV_NOISE = "Outdated Optimize Dep"

errors: list[str] = []
wanted = sys.argv[1:] or list(PAGES)
print("testing", BASE)
with sync_playwright() as p:
    browser = p.chromium.launch()
    for name in wanted:
        path, ready = PAGES[name]
        for view, width, height, mode in VIEWS:
            ctx = browser.new_context(viewport={"width": width, "height": height}, color_scheme=mode)
            ctx.add_init_script(f"localStorage.setItem('last_compiled_theme', 'dark'); "
                                f"localStorage.setItem('theme', '{mode}');")
            page = ctx.new_page()
            page.on("console", lambda m, n=f"{name}/{view}": errors.append(f"{n}: {m.text}")
                    if m.type == "error" and DEV_NOISE not in m.text else None)
            started = time.perf_counter()
            page.goto(BASE + path)
            page.locator(ready).first.wait_for(timeout=120_000)
            page.wait_for_timeout(900)   # fonts and the map's opening animation
            wide = page.evaluate("document.documentElement.scrollWidth")
            print(f"{name}/{view}: ready after {time.perf_counter() - started:.1f} s, page width {wide} px")
            if wide > width:
                errors.append(f"{name}/{view}: page is {wide} px wide on a {width} px screen")
            page.screenshot(path=f"screenshots/{name}-{view}.png", full_page=True)
            ctx.close()

    if "assistant" in wanted:   # the chat: each question gets an answer that says where it comes from
        for view, width, height, mode in VIEWS:
            if view.endswith("light") and view.startswith("phone"):
                continue
            ctx = browser.new_context(viewport={"width": width, "height": height}, color_scheme=mode)
            ctx.add_init_script(f"localStorage.setItem('last_compiled_theme', 'dark'); "
                                f"localStorage.setItem('theme', '{mode}');")
            page = ctx.new_page()
            page.on("console", lambda m, n=f"conversation/{view}": errors.append(f"{n}: {m.text}")
                    if m.type == "error" and DEV_NOISE not in m.text else None)
            page.goto(BASE + "/assistant")
            page.get_by_role("button", name=CONVERSATION[0]).click()
            for i, question in enumerate(CONVERSATION):
                if i:
                    page.get_by_label("Your question").fill(question)
                    page.get_by_label("Ask").click()
                page.locator(".msg.bot").nth(i).wait_for(timeout=120_000)
                if view == "desktop-dark":
                    row = page.locator(".msg.bot .source-row").nth(i).inner_text().replace("\n", " | ")
                    print(f"assistant: {question!r} – {row}")
            page.wait_for_timeout(500)
            # the example questions stay during the conversation
            if page.locator(".more-topics .example").count() != 8:
                errors.append(f"conversation/{view}: the example questions are gone after the first answer")
            page.screenshot(path=f"screenshots/assistant-conversation-{view}.png", full_page=True)
            ctx.close()
    browser.close()

print("problems:", errors or "none")
sys.exit(1 if errors else 0)
