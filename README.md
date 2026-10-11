# comp_eng
M.Sc. Space Engineering, Computational Methods

Course page: https://leibniz-iwt.github.io/comp_eng/


# Contributing

## Initial setup steps (requires conda):
1. git clone the repository to a local branch.
2. Set up a local conda environment using environment.yml with
Run `conda env create -f environment.yml`.

## The workflow for this repository is as follows:
1. Activate the environment with `conda activate compeng`.
2. Make changes to the `/notebooks` files. Helper code (plotting, widgets, long simulations) lives in the
   `notebooks/compeng/` package so that the notebooks only show the essential code; see `notebooks/compeng/README.md`.
3. When adding a new notebook, do _not_ manually add it, but add it to _toc.yml_ instead.
4. Run and validate the notebooks you changed. The website is built from the stored outputs, so this step
   is what the students will see:
   ```bash
   python tools/check_notebooks.py notebooks/03_chaos.ipynb --execute   # one notebook
   python tools/check_notebooks.py --toc                                 # every notebook in _toc.yml
   ```
   The checker also verifies the slide metadata: every cell carries a RISE slide type so the notebook can be
   presented with `jupyterlab-rise` (`Alt+R` in JupyterLab) and is exported to static slides by the website build
   (`python tools/build_slides.py`).
5. Commit and push your changes to the repository:
   ```bash
   git add .
   git commit -m "Your commit message"
   git push origin main
   ```
6. The changes will be automatically reflected on the GitHub pages site.

# Launch Individual Project Notebooks in Colab

For project notebooks (RL-based orbit station-keeping) on Colab:

- **[RL Station Keeping Handout](https://colab.research.google.com/github/Leibniz-IWT/comp_eng/blob/main/project_handout/RL_Station_Keeping_Handout.ipynb)**
- **[Sub-Problem 1](https://colab.research.google.com/github/Leibniz-IWT/comp_eng/blob/main/project_handout/RL_Station_Sub_Problem_1.ipynb)**
- **[Sub-Problem 2](https://colab.research.google.com/github/Leibniz-IWT/comp_eng/blob/main/project_handout/RL_Station_Sub_Problem_2.ipynb)**
