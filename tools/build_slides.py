#!/usr/bin/env python
"""Export every notebook listed in _toc.yml to a static reveal.js slideshow.

The slides use the RISE cell metadata (``slideshow.slide_type``) that is also
used for live presentation with jupyterlab-rise. Cells tagged ``hide-input``
have their source removed, exactly as the Jupyter Book hides them. Outputs are
taken from the notebook file (no execution), so run the notebooks first.

Usage (from the repository root)::

    python tools/build_slides.py --out _build/html/slides

The GitHub Actions workflow runs this after ``jupyter-book build`` so the slides
are served at https://leibniz-iwt.github.io/comp_eng/slides/<notebook>.slides.html
"""

from __future__ import annotations

import argparse
import html
import pathlib
import re
import sys

import nbformat
import yaml
from nbconvert import SlidesExporter
from traitlets.config import Config

ROOT = pathlib.Path(__file__).resolve().parents[1]
REVEAL_URL_PREFIX = "https://unpkg.com/reveal.js@4.6.1"
# Relative image/media paths in the notebooks resolve against notebooks/ (where the
# workflow copies them), but the slides are served from slides/: point them there.
_REL_ASSET = re.compile(r'(src|href)="(?:\./)?((?:images|media|cache)/)')


def relocate_assets(body: str, prefix: str = "../notebooks/") -> str:
    """Rewrite src="images/..." style paths so they work from the slides folder."""
    return _REL_ASSET.sub(lambda m: f'{m.group(1)}="{prefix}{m.group(2)}', body)


def toc_entries() -> list[tuple[pathlib.Path, str]]:
    toc = yaml.safe_load((ROOT / "_toc.yml").read_text())
    entries: list[tuple[pathlib.Path, str]] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            if "file" in node:
                p = ROOT / f"{node['file']}.ipynb"
                if p.exists() and p.parent.name == "notebooks":
                    entries.append((p, node.get("title", p.stem)))
            for key in ("chapters", "sections", "parts"):
                for child in node.get(key, []) or []:
                    walk(child)

    walk(toc)
    return entries


def make_exporter(reveal_url_prefix: str) -> SlidesExporter:
    c = Config()
    c.TagRemovePreprocessor.enabled = True
    c.TagRemovePreprocessor.remove_input_tags = {"hide-input", "remove-input"}
    c.TagRemovePreprocessor.remove_cell_tags = {"remove-cell"}
    c.TagRemovePreprocessor.remove_all_outputs_tags = {"remove-output"}
    c.SlidesExporter.exclude_input_prompt = True
    c.SlidesExporter.exclude_output_prompt = True
    c.SlidesExporter.reveal_url_prefix = reveal_url_prefix
    c.SlidesExporter.reveal_theme = "simple"
    c.SlidesExporter.reveal_transition = "none"
    c.SlidesExporter.reveal_scroll = True
    c.SlidesExporter.reveal_number = "c/t"
    return SlidesExporter(config=c)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "_build" / "html" / "slides")
    ap.add_argument("--reveal-url-prefix", default=REVEAL_URL_PREFIX)
    ap.add_argument("notebooks", nargs="*", type=pathlib.Path, help="override: only these notebooks")
    args = ap.parse_args()

    entries = [(p.resolve(), p.stem) for p in args.notebooks] if args.notebooks else toc_entries()
    if not entries:
        print("no notebooks found", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    exporter = make_exporter(args.reveal_url_prefix)
    index_rows: list[str] = []
    for path, title in entries:
        nb = nbformat.read(path, as_version=4)
        n_slides = sum(
            1 for c in nb.cells if c.get("metadata", {}).get("slideshow", {}).get("slide_type") in ("slide", "subslide")
        )
        body, _ = exporter.from_notebook_node(nb, resources={"metadata": {"name": path.stem}})
        body = relocate_assets(body)
        target = args.out / f"{path.stem}.slides.html"
        target.write_text(body, encoding="utf-8")
        print(f"{path.name:45s} -> {target.relative_to(ROOT) if target.is_relative_to(ROOT) else target}  ({n_slides} slides)")
        index_rows.append(f'<li><a href="{target.name}">{html.escape(title)}</a> <small>({n_slides} slides)</small></li>')

    index = (
        "<!doctype html><meta charset='utf-8'><title>Computational Methods: slides</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:48rem;margin:3rem auto;padding:0 1rem;line-height:1.5}</style>"
        "<h1>Computational Methods: lecture slides</h1>"
        "<p>Static reveal.js exports of the course notebooks. Use the arrow keys or swipe to navigate; "
        "press <kbd>Esc</kbd> for an overview. For live, executable slides open the notebook in JupyterLab "
        "and press <kbd>Alt</kbd>+<kbd>R</kbd> (RISE).</p>"
        f"<ol>{''.join(index_rows)}</ol>"
        "<p><a href='../intro.html'>Back to the course book</a></p>"
    )
    (args.out / "index.html").write_text(index, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
