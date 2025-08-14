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
2. Make changes to the `/notebooks` files and ensure that all the example code runs locally.
3. When adding a new notebook, do _not_ manually add it, but add it to _toc.yml_ instead.
4. Run `fix_books.py` (helper script to fix GitHub pages errors).
5. Commit and push your changes to the repository:
   ```bash
   git add .
   git commit -m "Your commit message"
   git push origin main
   ```
6. The changes will be automatically reflected on the GitHub pages site.