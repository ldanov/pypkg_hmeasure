#!/usr/bin/env python3

from setuptools import setup

setup(name='hmeasure',
      version='0.1',
      description='H-Measure Classification Metric',
      author='Lyubomir Danov',
      # author_email='-',
      url='https://github.com/ldanov/pypkg_hmeasure',
      packages=['hmeasure'],
      #TODO: find lower required versions
      install_requires = ['numpy == 1.16.3', 'scikit_learn == 0.21.2', 'scipy == 1.2.1']
     )