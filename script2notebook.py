from nbformat import v3, v4

with open('nnmammo.py') as script_file:
    script = script_file.read()

notebook = v4.writes(v4.upgrade(v3.reads_py(script))) + '\n'

with open('nnmammo.ipynb', 'w') as notebook_file:
    notebook_file.write(notebook)
