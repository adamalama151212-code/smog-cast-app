"""Build the assistant's knowledge base (the RAG index) from the files in this repo.

    docker compose --profile tools run --rm index                                   # build data/rag_index
    docker compose --profile tools run --rm index python tools/build_index.py --check      # is the index up to date?
    docker compose --profile tools run --rm index python tools/build_index.py --download   # fetch missing sources first

What goes in (conf/app.yaml, ``app.rag``): the documents written for the assistant (docs/rag/*.md), the official
sources as downloaded (data/rag_sources/, those not marked ``index: false``) and the gold data dictionary
(conf/gold_dictionary.yaml, one passage per result table). Change any of them and build again – the test
``test_the_index_matches_the_documents`` fails until you do.

Runs in the app's image: it has PyTorch and the embedding model; reading the PDF sources needs pymupdf, which the
``index`` service installs from requirements-dev.txt (the hosted app only searches the index and does not need it).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))   # run as a script from the repo root: tools/ is not a package

from smogcast.app import services  # noqa: E402
from smogcast.app.config import load_config  # noqa: E402
from smogcast.app.rag import Index, download_sources  # noqa: E402


def stale(cfg: dict, index_dir: Path) -> list[str]:
    """Why the saved index no longer matches the documents (empty = up to date). Compares the passages, which are
    cut without any model; the vectors follow from them and the embedding model named in the index."""
    if not (index_dir / "chunks.json").exists():
        return [f"no index at {index_dir}"]
    index = Index.load(index_dir)
    reasons = []
    if index.model_name != cfg["app"]["rag"]["embedding_model"]:
        reasons.append(f"built with {index.model_name}, configuration says {cfg['app']['rag']['embedding_model']}")
    now = services.knowledge_chunks(cfg)
    if now != index.chunks:
        changed = sorted({c.doc_id for c in now if c not in index.chunks} | {c.doc_id for c in index.chunks if c not in now})
        reasons.append(f"{len(index.chunks)} passages in the index, {len(now)} in the documents; differ in: {changed}")
    return reasons


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Build the smogcast knowledge base (RAG index).")
    parser.add_argument("--download", action="store_true", help="fetch the official sources that are missing first")
    parser.add_argument("--check", action="store_true", help="only tell whether the index matches the documents")
    parser.add_argument("--out", type=Path, default=services.INDEX_DIR, help="where to write the index")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config()
    rag = cfg["app"]["rag"]
    if args.download:
        present = download_sources(rag["sources"], ROOT / rag["sources_dir"])
        print(f"sources present: {len(present)} of {len(rag['sources'])}")
    if args.check:
        reasons = stale(cfg, args.out)
        print("the index matches the documents" if not reasons else "the index is out of date: " + "; ".join(reasons))
        return 1 if reasons else 0
    index = services.build_index(cfg, args.out)
    print(f"built {len(index.chunks)} passages -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
