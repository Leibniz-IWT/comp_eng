# `compeng` helper package

Supporting Python code for the course notebooks. One module per notebook:

| Module | Used by |
|--------|---------|
| `plotting.py` | all notebooks (shared figure style) |
| `chNN_<topic>.py` | lecture notebook `NN_<topic>.ipynb` |
| `tutNN_<topic>.py` | tutorial notebook `NN_tutorials-<topic>.ipynb` |

The notebooks import from these modules so that only the code that carries a
concept is shown inline. Everything else (plotting, widget wiring, long
simulations) lives here, where you can read it with syntax highlighting on
GitHub or open it next to the notebook in JupyterLab.

The notebooks are executed with the `notebooks/` directory as the working
directory, which is why `import compeng` works without installing anything.

`compeng.plotting.setup()` switches the grid on for every figure. If you add your
own plots, use `plt.grid(False)` to switch it off; a bare `plt.grid()` toggles the
grid and therefore switches it off.
