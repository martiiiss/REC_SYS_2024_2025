# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(
        "FunkSVD_fastest.pyx", 
        compiler_directives={'language_level': "3"}  # Compatibilità con Python 3
    ),
    include_dirs=[numpy.get_include()]
)