# -*- coding: UTF-8 -*-
"""
此脚本将使用Cython编译viterbi.pyx
"""


import os

from Cython.Build import cythonize
from distutils.core import setup
import numpy


# Windows下的存储路径与Linux并不相同
# 在Windows下使用Cython请参考https://github.com/cython/cython/wiki/InstallingOnWindows
if os.name == "nt":
    setup(
        name = "yahmm",
        description="Yet another hmm implimentation for supervised",
        packages=["yahmm"],
        ext_modules = cythonize(["hmm\\utils\\viterbi.pyx"]),
        requires=["sklearn"],
        include_dirs=[numpy.get_include()])
else:
    setup(
        name = "yahmm",
        description="Yet another hmm implimentation for supervised",
        packages=["yahmm"],
        ext_modules = cythonize(["hmm/utils/viterbi.pyx"]),
        requires=["sklearn"],
        include_dirs=[numpy.get_include()])
