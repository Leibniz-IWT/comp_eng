#!/usr/bin/env python
"""Mark the widget controls stored in a notebook's saved widget state as disabled.

The notebooks store their ipywidgets state so that the static website shows each
widget's initial figure. When such a notebook is opened in JupyterLab (locally or on
Binder), the stored widgets are restored before any cell has run; they look live but
have no Python object behind them, so moving a slider does nothing until the cell is
executed. Disabling the stored controls makes that visible: the restored sliders are
greyed out, and running the cell replaces them with live, enabled ones. Only the
notebook metadata is touched; outputs are unchanged.

Usage::

    python tools/disable_saved_widgets.py notebooks/*.ipynb
"""

from __future__ import annotations

import json
import pathlib
import sys

STATE_KEY = "application/vnd.jupyter.widget-state+json"
CONTROL_SUFFIXES = ("SliderModel", "DropdownModel", "CheckboxModel", "ToggleButtonsModel", "ButtonModel",
                    "TextModel", "IntTextModel", "FloatTextModel", "BoundedIntTextModel", "BoundedFloatTextModel",
                    "RadioButtonsModel", "SelectModel", "SelectionSliderModel", "ToggleButtonModel", "PlayModel")
TOOLTIP = "Run the cell above to activate this control (needs a running kernel: JupyterLab or Binder)"


def patch(path: pathlib.Path) -> int:
    nb = json.loads(path.read_text(encoding="utf-8"))
    state = nb.get("metadata", {}).get("widgets", {}).get(STATE_KEY, {}).get("state", {})
    n = 0
    for model in state.values():
        if model.get("model_name", "").endswith(CONTROL_SUFFIXES):
            st = model.setdefault("state", {})
            if not st.get("disabled"):
                st["disabled"] = True
                n += 1
            if "tooltip" in st and not st["tooltip"]:
                st["tooltip"] = TOOLTIP
    if n:
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return n


def main() -> int:
    total = 0
    for arg in sys.argv[1:]:
        p = pathlib.Path(arg)
        n = patch(p)
        total += n
        print(f"{p.name:45s} {n} control(s) disabled in the saved state")
    return 0 if total >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
