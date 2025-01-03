# setup.py
from setuptools import setup
from Cython.Build import cythonize
import numpy as np

# Elenco dei file Cython da compilare
cython_modules = [
    "SLIM_BPR_Cython_Epoch.pyx",
    "Sparse_Matrix_Tree_CSR.pyx",
    "Triangular_Matrix.pyx"
]

# Funzione di configurazione
setup(
    ext_modules=cythonize(
        cython_modules,
        compiler_directives={'language_level': "3"}  # Specifica il livello di Python, opzionale
    ),
    include_dirs=[np.get_include()],  # Include le directory di numpy
)