#to build, use:  python setup.py build_ext --inplace 

from distutils.core import setup
from Cython.Build import cythonize
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules = [
            Extension("cythonFactorize",
            ["cythonFactorize.pyx"],
            include_dirs=[np.get_include()])
                  ]
  #name = 'SGD Factorization',
  #ext_modules = cythonize("test.pyx"),
)