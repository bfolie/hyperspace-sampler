#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 14:20:10 2018

@author: brendan
"""

INIT_STEP_SIZE = .01 # initial step size
MAX_STEPS = 7000 # maximum number of steps before program terminates
CHECK_STEPS = 100 # how frequently it checks to see if the samples have stabilized
MIN_ACCEPT_RATE = 0.2 # minimum move acceptance rate (any lower and step size becomes smaller)
MAX_ACCEPT_RATE = 0.8 # maximum move acceptance rate (any higher and step size becomes bigger)
STEP_SIZE_FACTOR = 0.5 # factor by which step_size changes (must be <= 1)
TOLERANCE = 0.01 # how small variations have to be in order for the sampler to be considered stabilized