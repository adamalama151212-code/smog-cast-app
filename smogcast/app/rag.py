"""Retrieval over the knowledge documents (RAG).

Everything the knowledge base is built from lives in this repo: the documents written for the assistant (Markdown,
docs/rag/) and the official sources as downloaded (data/rag_sources/: PDF regulations and directives, HTML pages of
GIOŚ, WHO and EEA; the list with addresses is conf/app.yaml, ``app.rag.sources``). ``tools/build_index.py`` turns
them into the index (data/rag_index/). Every chunk remembers WHERE it comes from (document, page or section), so an
answer can cite it.

The index is a plain matrix of normalised embeddings: a few hundred chunks need no vector database.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Chunk:
    doc_id: str
    title: str
    publisher: str
    location: str        # "page 12" / "section: 6. How good the forecast is"
    text: str


# ----------------------------------------------------------------------------- downloading the official sources
# A browser-like agent: EUR-Lex and gov.pl serve plain clients the same documents, but some return an empty body to
# unknown agents.
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) smogcast/0.1"}


def looks_like(content: bytes, expect: str) -> bool:
    """Some portals answer automated clients with an HTML page (a captcha, a redirect loop) under a ``.pdf`` address:
    saving that as the "regulation" would silently poison the knowledge base."""
    head = content[:512].lstrip().lower()
    if expect == "pdf":
        return content.startswith(b"%PDF-")
    if expect == "html":
        return head.startswith((b"<!doctype html", b"<html")) and len(content) > 2000
    raise ValueError(f"Unknown expected format {expect!r}")


def download_sources(sources: list[dict], out_dir: str | Path, get=None) -> list[Path]:
    """Download the sources that are missing; return the paths of all sources present afterwards. Files already
    there are kept (delete one to fetch it again) – the originals in the repo stay the ones the index was built from."""
    if get is None:
        import requests  # only needed when a source is fetched

        get = requests.get
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    present = []
    for src in sources:
        target = out / src["file"]
        if target.exists():
            present.append(target)
            continue
        r = get(src["url"], headers=HEADERS, timeout=120)
        r.raise_for_status()
        if not looks_like(r.content, src["expect"]):
            raise ValueError(f"{src['id']}: expected {src['expect']} from {src['url']}, got something else "
                             f"({len(r.content)} bytes) – the portal may block automated downloads")
        tmp = target.with_suffix(target.suffix + ".part")
        tmp.write_bytes(r.content)
        tmp.replace(target)
        log.info("knowledge source downloaded: %s (%d kB)", target.name, len(r.content) // 1024)
        present.append(target)
    return present


# ----------------------------------------------------------------------------- loading
class _TextOnly(HTMLParser):
    """Visible text of an HTML page; navigation, scripts and forms are skipped (they only add noise)."""

    SKIP = {"script", "style", "noscript", "nav", "header", "footer", "form", "svg", "button", "select"}
    BLOCK = {"p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "section", "article", "br", "table"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip and data.strip():
            self.parts.append(data.strip() + " ")


def html_to_text(html: str) -> str:
    p = _TextOnly()
    p.feed(html)
    text = "".join(p.parts)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def clean_pdf_text(text: str) -> str:
    """Join words hyphenated at line ends and line breaks inside paragraphs (as in the RAG prototype)."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def markdown_sections(text: str) -> list[tuple[str, str]]:
    """(heading, body) per ``#``/``##``/``###`` section; the document title is kept in every chunk anyway."""
    sections, heading, body = [], "", []
    for line in text.splitlines():
        if re.match(r"^#{1,3} ", line):
            if "".join(body).strip():
                sections.append((heading, "\n".join(body).strip()))
            heading, body = line.lstrip("#").strip(), []
        else:
            body.append(line)
    if "".join(body).strip():
        sections.append((heading, "\n".join(body).strip()))
    return sections


def _tail(text: str, n: int) -> str:
    """The last ~n characters of a text, starting at a word boundary (a chunk never opens mid-word)."""
    if len(text) <= n:
        return text.strip()
    tail = text[-n:]
    return tail[tail.find(" ") + 1:].strip() if " " in tail else tail.strip()


# Where a repeated tail may begin: after the end of a sentence, at a new paragraph, at a list item.
_SENTENCE_START = re.compile(r"(?<=[.!?])[\"”’)*]*\s+|\n\s*\n|\n(?=\s*(?:[-*] |\d+\. ))")


def _carried(text: str, n: int) -> str:
    """What the next piece repeats from this one: whole sentences from its last ~n characters, and never part of a
    table. Half of anything misleads a language model that is asked to quote figures (found 2026-10-10 and -12):
    - the last rows of a Markdown table without the header row do not say what the columns mean ("490 of 667
      warnings" was read as the share of smog days caught);
    - the end of a sentence without its subject ("… on 388 of the 697 smog days") does not say whose result it is.
    A tail with no sentence start in it is not repeated at all."""
    after_the_last_table: list[str] = []
    for paragraph in reversed(text.split("\n\n")):
        if paragraph.lstrip().startswith("|"):
            break
        after_the_last_table.insert(0, paragraph)
    rest = "\n\n".join(after_the_last_table).strip()
    if len(rest) <= n:
        return rest                                  # whole paragraphs: they begin where a paragraph begins
    start = _SENTENCE_START.search(rest[-n:])
    return rest[-n:][start.end():].strip() if start else ""


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Paragraph-aware packing into pieces of about ``size`` characters. A piece repeats the last sentences of the
    previous one, up to ``overlap`` characters (see ``_carried``); only a paragraph longer than ``size`` – a PDF page
    without breaks – is cut inside, and there the overlap keeps a fact cut at the border findable."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces, current = [], ""
    for para in paragraphs:
        while len(para) > size:  # a single huge paragraph (e.g. a PDF page without breaks)
            cut = para.rfind(" ", 0, size) if para.rfind(" ", 0, size) > size // 2 else size
            if current:
                pieces.append(current)
                current = ""
            pieces.append(para[:cut].strip())
            para = _tail(para[:cut], overlap) + " " + para[cut:].strip()
        if current and len(current) + len(para) + 2 > size:
            pieces.append(current)
            carried = _carried(current, overlap) if overlap else ""
            current = f"{carried}\n\n{para}" if carried else para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        pieces.append(current)
    return pieces


def load_chunks(project_dir: str | Path, sources_dir: str | Path, sources: list[dict],
                size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for md in sorted(Path(project_dir).glob("*.md")):
        text = md.read_text(encoding="utf-8")
        title = next((line.lstrip("# ").strip() for line in text.splitlines() if line.startswith("# ")), md.stem)
        for heading, body in markdown_sections(text):
            for piece in split_text(body, size, overlap):
                chunks.append(Chunk(md.stem, title, "smogcast", f"section: {heading or title}", piece))
    for src in sources:
        path = Path(sources_dir) / src["file"]
        if not src.get("index", True):
            continue  # kept for provenance only (e.g. a Polish original whose English translation is indexed)
        if not path.exists():
            continue  # not downloaded yet – the index is built from what is available
        if src["expect"] == "pdf":
            import pymupdf  # only needed when building the index

            with pymupdf.open(path) as pdf:
                pages = [(f"page {i + 1}", clean_pdf_text(page.get_text())) for i, page in enumerate(pdf)]
        else:
            pages = [("web page", html_to_text(path.read_text(encoding="utf-8", errors="replace")))]
        for location, text in pages:
            for piece in split_text(text, size, overlap):
                if len(piece) >= 80:  # page numbers, running headers
                    chunks.append(Chunk(src["id"], src["title"], src["publisher"], location, piece))
    return chunks


# ----------------------------------------------------------------------------- embeddings and index
class Embedder:
    """sentence-transformers model, loaded lazily (hundreds of MB; only the assistant needs it)."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return np.asarray(self._model.encode(texts, normalize_embeddings=True, batch_size=32), dtype=np.float32)


def embed_text(c: Chunk) -> str:
    # The title and location travel with the text: "page 12" of a directive means nothing alone.
    return f"{c.title} – {c.location}\n{c.text}"


class Index:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, model_name: str):
        self.chunks, self.vectors, self.model_name = chunks, vectors, model_name

    @classmethod
    def build(cls, chunks: list[Chunk], embedder) -> Index:
        return cls(chunks, embedder.encode([embed_text(c) for c in chunks]), getattr(embedder, "model_name", "?"))

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vectors.npy", self.vectors)
        (d / "chunks.json").write_text(json.dumps({"model": self.model_name, "chunks": [asdict(c) for c in self.chunks]},
                                                  ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> Index:
        d = Path(directory)
        meta = json.loads((d / "chunks.json").read_text(encoding="utf-8"))
        return cls([Chunk(**c) for c in meta["chunks"]], np.load(d / "vectors.npy"), meta["model"])

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        scores = self.vectors @ query_vector.reshape(-1)
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]


def tokens(text: str) -> list[str]:
    """Lower-case words and numbers without Polish diacritics; "PM2,5" / "PM2.5" stay one token."""
    from smogcast.app.text import normalize

    return [t.replace(".", ",") for t in re.findall(r"[a-z]+[0-9]+(?:[.,][0-9]+)?|[0-9]+(?:[.,][0-9]+)?|[a-z]+",
                                                     normalize(text))]


class BM25:
    """Keyword relevance (Okapi BM25). Embeddings find paraphrases but blur exact terms – years, values,
    "poziom informowania" vs "poziom alarmowy"; keywords catch exactly those."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [tokens(t) for t in texts]
        self.lengths = np.array([len(d) for d in self.docs], dtype=np.float32)
        self.avg = float(self.lengths.mean()) if len(self.docs) else 1.0
        df: dict[str, int] = {}
        for d in self.docs:
            for term in set(d):
                df[term] = df.get(term, 0) + 1
        n = len(self.docs)
        self.idf = {term: float(np.log(1 + (n - f + 0.5) / (f + 0.5))) for term, f in df.items()}
        self.tf = [{term: d.count(term) for term in set(d)} for d in self.docs]

    def scores(self, query: str) -> np.ndarray:
        q = [t for t in set(tokens(query)) if t in self.idf]
        out = np.zeros(len(self.docs), dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.lengths / self.avg)
        for i, tf in enumerate(self.tf):
            out[i] = sum(self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm[i]) for t in q if t in tf)
        return out


class Retriever:
    """Hybrid search: reciprocal rank fusion of embedding similarity and BM25 keyword relevance."""

    RRF_K = 60  # standard constant: ranks matter, raw score scales of the two methods do not

    def __init__(self, index: Index, embedder, k: int):
        self.index, self.embedder, self.k = index, embedder, k
        self.bm25 = BM25([embed_text(c) for c in index.chunks])

    def __call__(self, question: str) -> list[tuple[Chunk, float]]:
        cosine = self.index.vectors @ self.embedder.encode([question])[0].reshape(-1)
        keyword = self.bm25.scores(question)
        fused = np.zeros(len(cosine), dtype=np.float32)
        for scores in (cosine, keyword):
            ranks = np.empty(len(scores), dtype=np.int64)
            ranks[np.argsort(-scores)] = np.arange(len(scores))
            fused += 1.0 / (self.RRF_K + ranks + 1)
        top = np.argsort(-fused)[: self.k]
        return [(self.index.chunks[i], float(cosine[i])) for i in top]
