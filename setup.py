import os
import re

from setuptools import setup

from hmeasure._version import __version__


def _parse_requirements(filepath):
    with open(filepath, mode='r') as fp:
        rqr_file = fp.readlines()

    rqr_cl = [x for x in rqr_file
              if not (re.match(r'^\s*$', x) or re.match(r'^#', x))]
    rqr = [re.sub('==', '~=', x.replace('\n', '')) for x in rqr_cl]

    return rqr


__PROJECT_NAME__ = 'hmeasure'
__DESCRIPTION__ = 'H-Measure Classification Metric'
__AUTHORS__ = 'Lyubomir Danov'
__URL__ = 'https://github.com/ldanov/pypkg_hmeasure'
__PACKAGES__ = [__PROJECT_NAME__]
req_loc = os.path.join(os.path.dirname(__file__), 'requirements.txt')
__INSTALL_REQUIRES__ = _parse_requirements(req_loc)

setup(name=__PROJECT_NAME__,
      version=__version__,
      description=__DESCRIPTION__,
      author=__AUTHORS__,
      url=__URL__,
      packages=__PACKAGES__,
      install_requires=__INSTALL_REQUIRES__
      )
