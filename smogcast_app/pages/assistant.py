"""Assistant – the chat, answered by smogcast.app.assistant with the conversation as history, so
"and in Wrocław?" is understood.

A welcome screen with four topics and examples; after the first question the topics stay, under the conversation
and above the question field; every answer says where it comes
from (forecast table, database query, documents, data dictionary) and shows its SQL, the rows of a data query and the
document passages it used. Source passages quote official documents verbatim and are shown through ``shown``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging

import reflex as rx

from smogcast.app import services
from smogcast.app.assistant import UNDERSTOOD
from smogcast.app.llm import LLMUnavailable
from smogcast_app.components.shell import layout, page_head, shown, sql_panel

log = logging.getLogger(__name__)

# Four topics with one or two examples each.
WELCOME = [
    ("cloud-fog", "Tomorrow's smog", ["Will there be smog in Kraków tomorrow?",
                                       "Why is the risk in Wrocław what it is tomorrow?"]),
    ("badge-check", "How good is the model", ["Why should I trust the PM2.5 forecast?",
                                              "Is the model better than “tomorrow like today”?"]),
    ("heart-pulse", "Health & limits", ["What should I do when the PM10 information level is exceeded?",
                                        "What are the EU limits for PM2.5 from 2030?"]),
    ("list-checks", "Data quality", ["How many readings were rejected and why?", "What does calm_hours mean?"]),
]
SOURCE = {"forecast": ("table-2", "From the forecast table"), "data": ("database", "From a database query"),
          "knowledge": ("book-open-text", "From the documents"), "dictionary": ("book-a", "From the data dictionary")}
MAX_ROWS_SHOWN = 50


@dataclasses.dataclass
class Passage:
    title: str = ""
    location: str = ""
    score: str = ""
    text: str = ""


@dataclasses.dataclass
class Message:
    role: str = "user"
    text: str = ""
    icon: str = ""
    source: str = ""
    sql: str = ""
    notes: list[str] = dataclasses.field(default_factory=list)
    columns: list[str] = dataclasses.field(default_factory=list)
    rows: list[list[str]] = dataclasses.field(default_factory=list)
    row_count: str = ""
    passages: list[Passage] = dataclasses.field(default_factory=list)
    model: str = ""       # the language model that wrote the answer; "" = written from the tables or the dictionary
    understood: str = ""  # the language model that only rewrote the question of an answer it did not write


# Answers the language model writes: from the documents and from a database query (it writes the SQL and the words,
# the numbers come from the rows). Forecast and dictionary answers are templates over the tables (no model).
MODEL_ROUTES = ("knowledge", "data")
WITHOUT_MODEL = "answered without the language model"   # the assistant's note when it fell back to passages


def cell(x) -> str:
    return "–" if x is None else str(x)


def from_answer(a, model: str = "") -> Message:
    icon, label = SOURCE.get(a.route, ("info", ""))
    columns = list(a.rows[0].keys()) if a.rows and a.route == "data" else []
    written = model if a.route in MODEL_ROUTES and WITHOUT_MODEL not in a.notes else ""
    # "And in Wrocław?": the numbers and the words come from the forecast table, but which city and which day was the
    # model's reading of the conversation – said once, and not next to "Written by the language model".
    understood = model if not written and any(n.startswith(UNDERSTOOD) for n in a.notes) else ""
    return Message(
        role="assistant", text=shown(a.text), icon=icon, source=label, sql=a.sql or "",
        notes=[shown(n) for n in a.notes],
        columns=columns, rows=[[cell(r.get(c)) for c in columns] for r in (a.rows or [])[:MAX_ROWS_SHOWN]]
        if columns else [], row_count=f"{len(a.rows)} rows" if columns else "",
        passages=[Passage(shown(c.title), shown(c.location), f"{score:.2f}", shown(c.text[:900]))
                  for c, score in a.sources],
        model=written, understood=understood)


def ask(question: str, history: list[dict]) -> Message:
    available, name = services.llm_status()
    return from_answer(services.shared_assistant().ask(question, history), name if available else "")


class AssistantState(rx.State):
    messages: list[Message] = []
    thinking: bool = False
    llm: str = ""

    @rx.event(background=True)
    async def open(self):
        ok, name = await asyncio.to_thread(services.llm_status)
        async with self:
            self.llm = (f"Language model: {name}" if ok else "The language model is not available: forecasts, the data "
                        "dictionary and document passages still work; questions about other numbers need it.")

    @rx.event
    def submit(self, form: dict):
        question = (form.get("question") or "").strip()
        if question:
            return AssistantState.send(question)

    @rx.event(background=True)
    async def send(self, question: str):
        async with self:
            if self.thinking:
                return
            history = [{"role": m.role, "content": m.text} for m in self.messages]
            self.messages = [*self.messages, Message(role="user", text=question)]
            self.thinking = True
        try:
            reply = await asyncio.to_thread(ask, question, history)
        except LLMUnavailable:
            # a hosted model now and then is busy or over its plan's limit, even after the retries (services.LLM_RETRIES)
            log.warning("the language model did not answer", exc_info=True)
            reply = Message(role="assistant", icon="cloud-off", source="Language model not answering",
                            text="The language model is not answering right now. Please ask again in a moment – "
                                 "forecasts, the data dictionary and the pages with ready-made numbers work without it.")
        except Exception:
            log.exception("the assistant failed")
            reply = Message(role="assistant", icon="cloud-off", source="Error",
                            text="Something went wrong while looking for the answer. Please try again in a minute.")
        async with self:
            self.messages = [*self.messages, reply]
            self.thinking = False

    @rx.event
    def clear(self):
        self.messages = []


def topic(icon: str, title: str, examples: list[str], heading=rx.el.h2) -> rx.Component:
    """One topic card; its examples are disabled while an answer is being looked for (a click would be ignored)."""
    return rx.el.article(
        heading(rx.icon(icon, size=18), title, class_name="topic-title"),
        *[rx.el.button(ex, on_click=AssistantState.send(ex), type="button", disabled=AssistantState.thinking,
                       class_name="example") for ex in examples],
        class_name="card topic")


def more_topics() -> rx.Component:
    """The same topics during a conversation, between the last answer and the question field."""
    return rx.el.section(
        rx.el.h2("Try another question", class_name="topics-title"),
        rx.el.div(*[topic(i, t, ex, rx.el.h3) for i, t, ex in WELCOME], class_name="topics"),
        class_name="more-topics")


def welcome() -> rx.Component:
    return rx.el.section(
        rx.el.p("Answers in English – you can ask in Polish too. Numbers come from the result tables, knowledge from "
                "official documents, and each answer says which.", class_name="lead narrow"),
        rx.el.div(*[topic(i, t, ex) for i, t, ex in WELCOME], class_name="topics"),
        class_name="welcome")


def rows_table(m) -> rx.Component:
    return rx.el.details(
        rx.el.summary(rx.icon("table-2", size=15), "Query result (", m.row_count, ")"),
        rx.el.div(rx.el.table(
            rx.el.thead(rx.el.tr(rx.foreach(m.columns, lambda c: rx.el.th(c)))),
            rx.el.tbody(rx.foreach(m.rows, lambda r: rx.el.tr(rx.foreach(r, lambda x: rx.el.td(x))))),
            class_name="table"), class_name="table-wrap"),
        class_name="disclosure")


def passages(m) -> rx.Component:
    return rx.el.details(
        rx.el.summary(rx.icon("book-open-text", size=15), "Sources (", m.passages.length(), ")"),
        rx.el.ol(rx.foreach(m.passages, lambda p: rx.el.li(
            rx.el.p(rx.el.strong(p.title), rx.el.span(p.location, class_name="passage-meta"),
                    rx.el.span("relevance ", p.score, class_name="passage-meta")),
            rx.el.blockquote(p.text, class_name="passage"))), class_name="passages"),
        class_name="disclosure")


@rx.memo
def answer_text(text: rx.Var[str]) -> rx.Component:
    """Markdown as its own component: rendered inline in the list of messages, its hooks would be called once per
    answer and React stops on the changing number of hooks."""
    return rx.markdown(text, class_name="answer prose")


def bubble(m) -> rx.Component:
    return rx.cond(
        m.role == "user",
        rx.el.div(rx.el.p(m.text), class_name="msg user"),
        rx.el.article(
            rx.el.div(
                rx.el.span(rx.icon(m.icon, size=14), m.source, class_name="source"),
                rx.cond(m.model != "", rx.el.span(rx.icon("bot", size=14), "Written by the language model (", m.model,
                                                  ")", class_name="source"), rx.fragment()),
                rx.cond(m.understood != "", rx.el.span(rx.icon("bot", size=14), "Question understood by the language "
                                                       "model (", m.understood, ")", class_name="source"),
                        rx.fragment()),
                class_name="source-row"),
            answer_text(text=m.text),
            rx.cond(m.sql != "", sql_panel(m.sql, "SQL query (through the guardrails)"), rx.fragment()),
            rx.cond(m.columns.length() > 0, rows_table(m), rx.fragment()),
            rx.cond(m.passages.length() > 0, passages(m), rx.fragment()),
            rx.foreach(m.notes, lambda n: rx.el.p(rx.icon("info", size=14), n, class_name="note")),
            class_name="msg bot"),
    )


def composer() -> rx.Component:
    return rx.el.form(
        rx.el.input(name="question", placeholder="Ask about tomorrow's smog, the model, the data or air-quality limits…",
                    aria_label="Your question", auto_complete="off", class_name="ask-input"),
        rx.el.button(rx.icon("send-horizontal", size=18), type="submit", class_name="send", aria_label="Ask"),
        on_submit=AssistantState.submit, reset_on_submit=True, class_name="composer")


def page() -> rx.Component:
    return layout(
        "/assistant",
        page_head("Assistant", "Ask about the forecast, the quality of the model and the data, or about air-quality "
                               "standards and health advice.",
                  rx.cond(AssistantState.llm != "", rx.el.p(rx.icon("bot", size=14), AssistantState.llm,
                                                            class_name="note"), rx.fragment())),
        rx.cond(AssistantState.messages.length() == 0, welcome(),
                rx.el.div(rx.foreach(AssistantState.messages, bubble),
                          rx.el.button(rx.icon("eraser", size=14), "Start a new conversation",
                                       on_click=AssistantState.clear, type="button", class_name="link-button"),
                          class_name="thread", aria_live="polite")),
        rx.cond(AssistantState.thinking,
                rx.el.p(rx.icon("loader-circle", size=16, class_name="spin"), "Looking for the answer…",
                        class_name="note thinking", role="status"), rx.fragment()),
        rx.cond(AssistantState.messages.length() == 0, rx.fragment(), more_topics()),
        composer(),
    )
