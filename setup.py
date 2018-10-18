#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 18:16:13 2018

@author: brendan
"""

from setuptools import setup

setup(name = "hyperspace_sampler",
      version = "1.0",
      description = "Randomly samples a high-dimensional space with constraints",
      author = 'Brendan Folie',
      author_email = "bfolie@berkeley.edu",
      url = "https://github.com/bfolie/hyperspace-sampler",
      packages = [''],
      include_package_data=True,
      install_requires = ['numpy']
)