#!/usr/bin/env python3

from setuptools import setup
from hmeasure._version import __version__

__PROJECT_NAME__ = 'hmeasure'
__DESCRIPTION__ = 'H-Measure Classification Metric'
__AUTHORS__ = 'Lyubomir Danov'
__URL__ = 'https://github.com/ldanov/pypkg_hmeasure'
__PACKAGES__ = [__PROJECT_NAME__]

setup(name=__PROJECT_NAME__,
      version=__version__,
      description=__DESCRIPTION__,
      author=__AUTHORS__,
      # author_email='-',
      url=__URL__,
      packages=__PACKAGES__,
      #TODO: find lower required versions
      install_requires = ['numpy>=1.16.3', 'scikit_learn>=0.21.2', 'scipy>=1.2.1']
     )