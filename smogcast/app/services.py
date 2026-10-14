"""Wiring for the app: configuration, gold reader, language model and RAG index.

Everything comes from the configuration (config.py) and the environment – no pipeline context, no Spark.

RAG: the knowledge base is built in this repo, from this repo's files
(``knowledge_chunks``: docs/rag, data/rag_sources and the gold data dictionary; ``build_index`` / tools/build_index.py
→ data/rag_index), and searched with the same model and library – sentence-transformers on PyTorch, the model baked
into the image. The model loads in about 10 s, so ``warm_up`` runs when the server starts: nobody waits for it in
their first question.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

from smogcast.app import forecast as fc
from smogcast.app.assistant import Assistant
from smogcast.app.config import load_config
from smogcast.app.gold import GoldReader
from smogcast.app.llm import LLM
from smogcast.app.rag import Chunk, Embedder, Index, Retriever, load_chunks

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]   # the repo: docs/rag, data/rag_sources and data/rag_index hang off it
INDEX_DIR = Path(os.environ.get("SMOGCAST_RAG_INDEX", ROOT / "data" / "rag_index"))


def gold_reader(cfg: dict, max_rows: int | None = None, backend=None) -> GoldReader:
    return GoldReader(backend, max_rows or cfg["app"]["sql"]["max_rows"], dictionary=cfg.get("gold_dictionary", {}))


def dictionary_chunks(dictionary: dict) -> list[Chunk]:
    """The gold data dictionary (conf/gold_dictionary.yaml) as knowledge passages, one per result table: the
    same descriptions as in Unity Catalog and in the text-to-SQL prompt, so "why calm hours?" finds them too."""
    out = []
    for table, entry in dictionary.items():
        lines = [f"Result table {table}: {entry['description']}", "Columns:"]
        lines += [f"- {c}: {s['description']}" for c, s in entry["columns"].items()]
        out.append(Chunk("gold-data-dictionary", "smogcast result tables: what each column means", "smogcast",
                         f"table: {table}", "\n".join(lines)))
    return out


def knowledge_chunks(cfg: dict) -> list[Chunk]:
    """Every passage of the knowledge base, cut from the files in this repo – no model involved, so a test can tell
    whether the saved index still matches its documents. Reading the PDF sources needs pymupdf (requirements-dev)."""
    rag = cfg["app"]["rag"]
    chunks = load_chunks(ROOT / rag["project_docs_dir"], ROOT / rag["sources_dir"], rag["sources"],
                         rag["chunk_chars"], rag["overlap_chars"])
    return chunks + dictionary_chunks(cfg.get("gold_dictionary", {}))


def build_index(cfg: dict, index_dir: Path = INDEX_DIR) -> Index:
    """(Re)build the index from the repo's documents and save it; run by tools/build_index.py in the app's image."""
    rag = cfg["app"]["rag"]
    chunks = knowledge_chunks(cfg)
    if not chunks:
        raise FileNotFoundError(f"No documents in {rag['project_docs_dir']} or {rag['sources_dir']}")
    index = Index.build(chunks, Embedder(rag["embedding_model"]))
    index.save(index_dir)
    by_doc: dict[str, int] = {}
    for c in chunks:
        by_doc[c.doc_id] = by_doc.get(c.doc_id, 0) + 1
    log.info("RAG index: %d chunks %s -> %s", len(chunks), by_doc, index_dir)
    return index


def load_retriever(cfg: dict, index_dir: Path = INDEX_DIR) -> Retriever | None:
    rag = cfg["app"]["rag"]
    if not (index_dir / "chunks.json").exists():
        log.warning("RAG index missing at %s – the assistant answers without the knowledge base", index_dir)
        return None
    index = Index.load(index_dir)
    if index.model_name != rag["embedding_model"]:
        log.warning("RAG index was built with %s, configuration says %s – rebuild it", index.model_name,
                    rag["embedding_model"])
    return Retriever(index, Embedder(rag["embedding_model"]), rag["top_k"])


LLM_RETRIES = 2   # a hosted model answers "503 high demand" now and then: retried with backoff


def llm_client(llm: dict):
    """The OpenAI-compatible client of ``LLM``, but with retries: its default client has ``max_retries=0``
    (fine for LM Studio on the same machine), and one busy moment of a hosted model would end the answer."""
    from openai import OpenAI

    return OpenAI(base_url=llm["base_url"], api_key=llm["api_key"], timeout=llm["timeout_s"], max_retries=LLM_RETRIES)


def build_assistant(cfg: dict | None = None, reader: GoldReader | None = None) -> Assistant:
    cfg = cfg or load_config()
    llm = cfg["app"]["llm"]
    return Assistant(cfg, reader or gold_reader(cfg), LLM(llm, client=llm_client(llm)), load_retriever(cfg))


_shared: Assistant | None = None
_lock = threading.Lock()


def shared_assistant() -> Assistant:
    """One assistant per app process, built on first use (the page says "Preparing the assistant…") or by
    ``warm_up``."""
    global _shared
    with _lock:
        if _shared is None:
            _shared = build_assistant()
        return _shared


def warm_up() -> None:
    """Build the assistant and load the embedding model now, instead of during the first knowledge question."""
    assistant = shared_assistant()
    if assistant.retriever is not None:
        assistant.retriever.embedder.encode(["warm up"])
        log.info("assistant ready: %d knowledge passages", len(assistant.retriever.index.chunks))


# ------------------------------------------------------------------------------------------- shared gold data
# One read of the five page queries per process, shared by every visitor, read again after 5 minutes or when someone
# presses "Refresh data".
GOLD_TTL_S = 300
_gold: dict = {"at": 0.0, "data": None}
_gold_lock = threading.Lock()


def gold_data(refresh: bool = False) -> dict:
    """forecast.load(): forecast, reliability, metrics, acceptance, dq – each a QueryResult (rows + SQL + source)."""
    with _gold_lock:
        if refresh or _gold["data"] is None or time.monotonic() - _gold["at"] > GOLD_TTL_S:
            cfg = load_config()
            _gold["data"] = fc.load(gold_reader(cfg, max_rows=1000))
            _gold["at"] = time.monotonic()
        return _gold["data"]


_llm_status: dict = {"at": 0.0, "value": (False, "")}


def llm_status() -> tuple[bool, str]:
    """(available, model name or reason), checked at most every 30 s."""
    if time.monotonic() - _llm_status["at"] > 30:
        llm = shared_assistant().llm
        try:
            _llm_status["value"] = (True, llm.model)
        except Exception as exc:   # LLMUnavailable or a configuration error: shown as "unavailable"
            _llm_status["value"] = (False, str(exc).split(":")[0])
        _llm_status["at"] = time.monotonic()
    return _llm_status["value"]
