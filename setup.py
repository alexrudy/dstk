#! /usr/bin/env python
# -*- coding: utf-8 -*-

import DSTK
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy
import os
import platform

extensions = [
    Extension(name="DSTK.Timeseries._recurrence_map",
              sources=['DSTK/Timeseries/_recurrence_map.pyx'],
              include_dirs=[numpy.get_include()],
              extra_compile_args=['-fopenmp'],
              extra_link_args=['-fopenmp'])
              ]

setup(
    author="Johannes Otterbach",
    author_email="johannesotterbach@gmail.com",
    name="DSTK",
    packages=find_packages(),
    version=DSTK.__version__,
    zip_safe=False,
    description="Data Science Toolkit",
    url='https://github.com/jotterbach/dstk',
    install_requires=[
        'numpy',
        'scipy',
        'tensorflow',
        'scikit-learn',
        'pandas',
        'fuzzywuzzy',
        'statsmodels',
        'patsy',
        'cython'
    ],
    tests_require=[
        'pytest',
        'cython'
    ],
    test_suite='DSTK.tests',
    dependency_links=[
        "https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.12.0rc0-py2-none-any.whl"
    ],
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(extensions)
)
