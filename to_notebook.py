from nbformat import v3, v4

with open("mnist.py") as f:
    text = f.read()

nb = v3.reads_py(text)
nb = v4.upgrade(nb)

with open("binary-net-mnist.ipynb", "w") as f:
    f.write(v4.writes(nb))
