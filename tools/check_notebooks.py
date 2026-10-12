#!/usr/bin/env python
"""Validate (and optionally execute) course notebooks.

Checks the authoring conventions described in notebooks/CLAUDE.md:

* every cell carries a RISE ``slideshow.slide_type``;
* the first cell is a level-1 title;
* markdown cells are short enough for one slide (warning);
* visible code cells are short (warning), long code must be tagged ``hide-input``;
* no empty code cells, no error outputs, no code cell without outputs
  (the site is built with ``execute_notebooks: off`` so stored outputs are
  what the students see);
* notebook file size stays within the budget.

With ``--execute`` the notebook is run in place (working directory =
notebook directory, widget state stored) before the checks.

Usage::

    python tools/check_notebooks.py notebooks/03_chaos.ipynb [--execute]
    python tools/check_notebooks.py --toc            # all notebooks in _toc.yml
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import nbformat

ROOT = pathlib.Path(__file__).resolve().parents[1]
SLIDE_TYPES = {"slide", "subslide", "fragment", "skip", "notes"}
MAX_MD_WORDS = 160
MAX_VISIBLE_CODE_LINES = 20
MAX_NB_MB = 3.0
WARN_NB_MB = 2.0


def toc_notebooks() -> list[pathlib.Path]:
    import yaml

    toc = yaml.safe_load((ROOT / "_toc.yml").read_text())
    files: list[pathlib.Path] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            if "file" in node:
                p = ROOT / f"{node['file']}.ipynb"
                if p.exists():
                    files.append(p)
            for key in ("chapters", "sections", "parts"):
                for child in node.get(key, []) or []:
                    walk(child)

    walk(toc)
    return files


def _make_client_class():  # noqa: ANN202
    """Return a NotebookClient subclass that does not stall on ipywidgets cells.

    nbclient 0.10 mirrors every Output-widget message back to the kernel on the
    shell socket while it is awaiting the execute reply on that same socket
    (``OutputWidget.sync_state``). The reply can then arrive without waking the
    poller, and the client waits for the whole cell timeout before noticing.
    Two remedies: the echo is switched off (the widget state is still stored,
    ``OutputWidget.output`` writes it directly), and the reply is awaited in
    one-second slices so a missed wake-up costs at most a second.
    """
    import asyncio
    import queue

    from nbclient import NotebookClient, output_widget
    from nbclient.exceptions import CellTimeoutError

    output_widget.OutputWidget.sync_state = lambda self: None  # type: ignore[method-assign]

    class SlicedClient(NotebookClient):
        async def _async_poll_for_reply(  # type: ignore[override]
            self, msg_id, cell, timeout, task_poll_output_msg, task_poll_kernel_alive
        ):
            deadline = None if timeout is None else time.monotonic() + timeout
            while True:
                try:
                    msg = await self.kc.shell_channel.get_msg(timeout=1.0)
                except queue.Empty:
                    if deadline is not None and time.monotonic() > deadline:
                        task_poll_kernel_alive.cancel()
                        raise CellTimeoutError.error_from_timeout_and_cell(
                            "Cell execution timed out", timeout, cell
                        ) from None
                    continue
                if msg["parent_header"].get("msg_id") == msg_id:
                    try:
                        await asyncio.wait_for(task_poll_output_msg, self.iopub_timeout)
                    except (asyncio.TimeoutError, queue.Empty):
                        self.log.warning("Timeout waiting for IOPub output")
                    task_poll_kernel_alive.cancel()
                    return msg

    return SlicedClient


def execute(path: pathlib.Path, timeout: int) -> list[str]:
    """Execute ``path`` in place; return a list of problems."""
    nb = nbformat.read(path, as_version=4)
    client = _make_client_class()(
        nb,
        timeout=timeout,
        kernel_name=nb.metadata.get("kernelspec", {}).get("name", "python3"),
        allow_errors=True,
        store_widget_state=True,
        coalesce_streams=True,
        resources={"metadata": {"path": str(path.parent)}},
    )
    t0 = time.perf_counter()
    client.execute()
    dt = time.perf_counter() - t0
    nbformat.write(nb, path)
    problems: list[str] = []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                problems.append(f"cell {i}: {out.get('ename')}: {str(out.get('evalue'))[:200]}")
    print(f"  executed in {dt:.1f} s")
    return problems


def check(path: pathlib.Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    nb = nbformat.read(path, as_version=4)
    try:
        nbformat.validate(nb)
    except nbformat.ValidationError as exc:  # pragma: no cover
        errors.append(f"invalid notebook: {exc}")

    size_mb = path.stat().st_size / 1e6
    if size_mb > MAX_NB_MB:
        errors.append(f"file size {size_mb:.1f} MB exceeds {MAX_NB_MB} MB (move media to notebooks/media/)")
    elif size_mb > WARN_NB_MB:
        warnings.append(f"file size {size_mb:.1f} MB above {WARN_NB_MB} MB target")

    cells = nb.cells
    if not cells or cells[0].cell_type != "markdown" or not cells[0].source.lstrip().startswith("# "):
        errors.append("first cell must be a markdown level-1 title")

    for i, cell in enumerate(cells):
        meta = cell.get("metadata", {})
        tags = set(meta.get("tags", []))
        slide = meta.get("slideshow", {}).get("slide_type")
        if slide not in SLIDE_TYPES:
            errors.append(f"cell {i}: missing/invalid slideshow.slide_type ({slide!r})")
        src = cell.source
        if cell.cell_type == "markdown":
            words = len(src.split())
            if words > MAX_MD_WORDS and "|" not in src:
                warnings.append(f"cell {i}: markdown has {words} words (> {MAX_MD_WORDS}); split for slides")
            if src.strip() == "":
                errors.append(f"cell {i}: empty markdown cell")
        elif cell.cell_type == "code":
            lines = [ln for ln in src.splitlines() if ln.strip()]
            if not lines:
                errors.append(f"cell {i}: empty code cell")
                continue
            if len(lines) > MAX_VISIBLE_CODE_LINES and not (tags & {"hide-input", "remove-input"}):
                warnings.append(
                    f"cell {i}: {len(lines)} visible code lines (> {MAX_VISIBLE_CODE_LINES}); move to compeng or tag hide-input"
                )
            if any(ln.strip().startswith(("if 0:", "if False:")) for ln in lines):
                errors.append(f"cell {i}: dead code block (if 0 / if False)")
            outputs = cell.get("outputs", [])
            if not outputs:
                warnings.append(f"cell {i}: code cell has no outputs (fine only if it produces none)")
            for out in outputs:
                if out.get("output_type") == "error":
                    errors.append(f"cell {i}: error output {out.get('ename')}")
                data = out.get("data", {})
                for mime in ("text/html", "image/png", "image/jpeg", "video/mp4"):
                    blob = data.get(mime)
                    if blob and len(json.dumps(blob)) > 1_000_000:
                        errors.append(f"cell {i}: {mime} output > 1 MB; save to notebooks/media/ instead")
    return errors, warnings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("notebooks", nargs="*", type=pathlib.Path)
    ap.add_argument("--toc", action="store_true", help="check every notebook listed in _toc.yml")
    ap.add_argument("--execute", action="store_true", help="execute in place before checking")
    ap.add_argument("--timeout", type=int, default=1200, help="per-cell timeout in seconds")
    ap.add_argument("-W", "--warnings-as-errors", action="store_true")
    args = ap.parse_args()

    paths = list(args.notebooks)
    if args.toc:
        paths += toc_notebooks()
    if not paths:
        ap.error("give notebook paths or --toc")

    failed = False
    for path in paths:
        path = path.resolve()
        print(f"== {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
        errors: list[str] = []
        if args.execute:
            errors += execute(path, args.timeout)
        e, w = check(path)
        errors += e
        for msg in w:
            print(f"  WARNING {msg}")
        for msg in errors:
            print(f"  ERROR   {msg}")
        if errors or (args.warnings_as_errors and w):
            failed = True
        if not errors and not w:
            print("  OK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
