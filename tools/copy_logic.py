"""Copy the DATA CONTRACT from the pipeline repo, unchanged, and record a SHA-256 of every file in
smogcast/app/COPIED.txt.

The contract is what the pipeline produces the gold tables by, so it is changed only there and copied again:
- conf/cities.yaml, conf/thresholds.yaml, conf/model.yaml – cities, limits and the model's settings,
- conf/gold_dictionary.yaml – the meaning of every gold column (also the comments, tags and masks in Unity Catalog),
- smogcast/core/schema.py – the model's input names (the Info page).

    python tools/copy_logic.py ["C:/Users/Leszek/DEMO RADPLUME/DEMO"]          # copy
    python tools/copy_logic.py --check [...]                                   # only compare, copy nothing

NOT copied – this repo's own since 2026-10-10: the assistant's logic (__init__, assistant, forecast, guardrails, llm,
rag, text), conf/app.yaml and, since 2026-10-11, the whole knowledge base: its documents (docs/rag), the downloaded
sources (data/rag_sources) and the index built from them (data/rag_index, tools/build_index.py). After a change of
gold_dictionary.yaml build the index again: the dictionary is part of the knowledge base (a test fails until you do).
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
CONF = ["cities.yaml", "thresholds.yaml", "model.yaml", "gold_dictionary.yaml"]
HEADER = (
    "# The data contract, copied unchanged from the pipeline repo by tools/copy_logic.py (SHA-256, paths from the\n"
    "# repo root): the pipeline produces the gold tables by these files. Do not edit them here: change them in the\n"
    "# pipeline repo and copy again. Everything else in smogcast/app (the assistant's logic and conf/app.yaml\n"
    "# included) and the knowledge base (docs/rag, data/rag_sources, data/rag_index) are this repo's own.\n"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pairs(pipeline: Path) -> list[tuple[Path, Path]]:
    core = pipeline / "packages/core/src/smogcast/core"
    return ([(core / "conf" / f, HERE / "smogcast/app/conf" / f) for f in CONF]
            # the model's input names for the Info page (no imports; the rest of smogcast-core is not needed)
            + [(core / "schema.py", HERE / "smogcast/core/schema.py")])


def main(argv: list[str]) -> int:
    check = "--check" in argv
    args = [a for a in argv if a != "--check"]
    pipeline = Path(args[0] if args else "C:/Users/Leszek/DEMO RADPLUME/DEMO")
    lines, differ = [], []
    for src, dst in pairs(pipeline):
        if check:
            if not dst.exists() or sha(src) != sha(dst):
                differ.append(str(dst.relative_to(HERE)))
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        lines.append(f"{sha(dst)}  {dst.relative_to(HERE).as_posix()}")
    if check:
        print("identical to the pipeline repo" if not differ else f"differ: {differ}")
        return 1 if differ else 0
    (HERE / "smogcast/app/COPIED.txt").write_text(HEADER + "\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
