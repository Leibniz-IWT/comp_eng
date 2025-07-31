import nbformat, pathlib
for nb_path in pathlib.Path("notebooks").rglob("*.ipynb"):
    nb = nbformat.read(nb_path, as_version=4)
    changed = False
    for c in nb.cells:
        if c.cell_type == "code" and "execution_count" not in c:
            c.execution_count = None
            changed = True
    if changed:
        nbformat.write(nb, nb_path)
        print("fixed", nb_path)