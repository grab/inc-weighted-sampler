# Copyright 2022 Grabtaxi Holdings Pte Ltd (GRAB), All rights reserved.

# Use of this source code is governed by an MIT-style license that can be found in the LICENSE file

from setuptools import setup
from setuptools import find_packages

setup(
    name='inc',
    version='0.0.1',
    install_requires=[
        'requests',
        'importlib-metadata; python_version >= "3.7"',
        'pyparsing >= 2.3.0',
        'gmpy2 == 2.0.8',
        'numpy >= 1.15.4',
        'psutil',
        'toposort == 1.6',
    ],
    # packages=find_packages(
    #     where='src',
    #     include=['mypackage*'],  # ["*"] by default
    #     exclude=['mypackage.tests'],  # empty by default
    # ),
)