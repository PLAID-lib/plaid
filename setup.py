# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#


from setuptools import setup

if __name__ == "__main__":

    setup()

# %% Imports

# from __future__ import absolute_import, print_function

# import datetime
# import os

# from setuptools import find_packages, setup

# # %% Globals

# name = 'plaid'
# version = '0.1'
# release = '0.1'
# author = "Safran"
# copyright = "2023-{}, {}".format(datetime.date.today().year, author)
# url = "https://github.com/PLAID-lib/plaid"

# # %% Script

# setup(name=name,
#       version=version,
#       description="A package that implements a data model tailored for AI and ML in the context of physics problems",
#       package_dir={"": "src"},
#       packages=find_packages(where="src"),
#       # packages=["src/plaid"],
#       author=author,
#       url=url,
#       #   entry_points={
#       #         "console_scripts": ["plaid = run:main"],
#       #         },
#       command_options={
#           'build_sphinx': {
#               'project': ('setup.py', name),
#               # 'author' : ('setup.py', author),
#               'copyright': ('setup.py', copyright),
#               'version': ('setup.py', version),
#               'release': ('setup.py', release),
#               'source_dir': ('setup.py', 'docs/source'),
#               'build_dir': ('setup.py', 'docs/_build'),
#           },
#       },
#       project_urls={
#           'Changelog': os.path.join(url, 'CHANGELOG.rst'),
#           'Issue Tracker': os.path.join(url, 'issues'),
#       },
#       keywords=[
#           'Data-Model',
#           'Safran',
#       ],
#       python_requires=">=3.11",
#       install_requires=[
#           # eg: 'aspectlib==1.1.1', 'six>=1.7',
#       ],
#       extras_require={
#           # eg:
#           #   'rst': ['docutils>=0.11'],
#           #   ':python_version=="2.6"': ['argparse'],
#       },
#       setup_requires=[
#           # 'pytest-runner',
#       ],
#       )
