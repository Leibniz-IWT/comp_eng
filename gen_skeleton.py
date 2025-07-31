#python - <<'PY'
import nbformat as nbf, pathlib, re

# Collect slugs straight from _toc.yml so you stay in sync
slugs = []
toc = pathlib.Path('_toc.yml').read_text()
for line in toc.splitlines():
    m = re.match(r'\s*-\s*file:\s*notebooks/([^\s]+)', line)
    if m:
        slugs.append(m.group(1))

path = pathlib.Path('notebooks')
path.mkdir(exist_ok=True)

for slug in slugs:
    nb_file = path / f'{slug}.ipynb'
    if nb_file.exists():
        continue
    nb = nbf.v4.new_notebook()
    title = slug.replace('_', ' ').title()
    nb['cells'] = [
        nbf.v4.new_markdown_cell(f'# {title}\n\n_Placeholder – content coming soon._')
    ]
    nb.metadata.update({"kernelspec": {"name": "python3", "display_name": "Python 3"}})
    nbf.write(nb, nb_file)
    print(f'created {nb_file}')
#PY
