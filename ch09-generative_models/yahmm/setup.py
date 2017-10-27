from Cython.Build import cythonize
from distutils.core import setup
import numpy


setup(
  name = "yahmm",
  description="Yet another hmm implimentation for supervised",
  packages=["yahmm"],
  ext_modules = cythonize(["hmm/utils/viterbi.pyx"]),
  requires=["sklearn"],
  include_dirs=[numpy.get_include()]
)
