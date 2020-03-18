#!/usr/bin/env python
from nbformat import v3, v4
import os
import sys

with open(sys.argv[1]) as f:
    text = f.read()

nb = v3.reads_py(text)
nb = v4.upgrade(nb)

with open(os.path.splitext(sys.argv[1])[0] + ".ipynb", "w") as f:
    f.write(v4.writes(nb))
