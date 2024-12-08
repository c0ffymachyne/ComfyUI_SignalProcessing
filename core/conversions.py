#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: C0ffymachyne
License: GPLv3
Version: 1.0.0

Description:
    various conversion methods
"""

import torch

def db_to_lin(value):
    return 10**(value / 20)

def lin_to_tb(value):
    return 20 * torch.log10(torch.abs(value) + 1.0e-24)

def get_sign(value):
    sign = torch.sign(value)
    return sign

