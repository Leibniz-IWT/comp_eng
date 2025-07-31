import pathlib, yaml, nbformat as nbf

def collect_files(node):
    files = []
    if isinstance(node, dict):
        if "file" in node:
            files.append(node["file"])
        for key in ("chapters", "sections"):
            for child in node.get(key, []):
                files.extend(collect_files(child))
    return files

toc = yaml.safe_load(pathlib.Path("_toc.yml").read_text())
slugs = collect_files(toc)

path = pathlib.Path("notebooks")
path.mkdir(exist_ok=True)

for slug in slugs:
    nb_file = pathlib.Path(f"{slug}.ipynb")
    if nb_file.exists():
        continue
    nb = nbf.v4.new_notebook()
    title = slug.split("/")[-1].replace("_", " ").title()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(f"# {title}\n\n_Placeholder – content coming soon._")
    ]
    nbf.write(nb, nb_file)
    print("created", nb_file)
