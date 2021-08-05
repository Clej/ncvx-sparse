#! /usr/bin/env python

import codecs
import os

from setuptools.command.build_ext import build_ext
from setuptools import Extension
from setuptools import find_packages, setup
from setuptools import dist
dist.Distribution().fetch_build_eggs(['numpy>=1.12'])
import numpy as np

# get __version__ from _version.py
ver_file = os.path.join('ncvxsp', '_version.py')
with open(ver_file) as f:
    exec(f.read())

DISTNAME = 'ncvx-sparse'
DESCRIPTION = 'Scikit-learn compatible implementation of nonconvex sparse estimators for single- and multi-task linear regressions (e.g. SCAD, MCP, l1-group-SCAD, etc).'
with codecs.open('README.rst', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'Clement Lejeune'
MAINTAINER_EMAIL = 'clement.lejeune@irit.fr'
URL = 'https://github.com/scikit-learn-contrib/Clej/ncvx_estimators'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/scikit-learn-contrib/Clej/ncvx_estimators'
VERSION = __version__
INSTALL_REQUIRES = ['numpy', 'scipy', 'scikit-learn>=0.23', 'cython>=0.26']
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7']
EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov'],
    'docs': [
        'sphinx',
        'sphinx-gallery',
        'sphinx_rtd_theme',
        'numpydoc',
        'matplotlib'
    ]
}

setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      zip_safe=False,  # the package can run out of an .egg file
      classifiers=CLASSIFIERS,
      packages=find_packages(),
      python_requires=">=3.6",
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE,
      cmdclass={'build_ext': build_ext},
      ext_modules=[
        Extension('ncvxsp.linear_model.scad_cd_fast',
                    sources=['ncvxsp/linear_model/scad_cd_fast.pyx'],
                    language='c',
                    include_dirs=[np.get_include()]
                    )
      ]

      )
