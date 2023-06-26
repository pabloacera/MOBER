#!/usr/bin/env python
# -*- coding: utf-8 -*-


from setuptools import setup, find_packages

setup(
    name='mober_2',
    version='2.0.1',
    url='ssh://git@bitbucket.prd.nibr.novartis.net/ods/ods-mober.git',
    description='Modified mober',
    packages=find_packages(),
    install_requires=['mlflow', 'scanpy'],
    
    keywords='mober',

    entry_points={
        'console_scripts': ['mober_2 = mober.mober:main']
    }
)
