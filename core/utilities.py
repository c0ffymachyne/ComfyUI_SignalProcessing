#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: C0ffymachyne
License: GPLv3
Version: 1.0.0

Description:
    various development utilities
"""

import os
import sys
import re

# Define the regex
pattern = r"^ComfyUI-\d+\.\d+\.\d+$"

this_file_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)))


def find_comfy_root():
    path = os.fspath(this_file_directory)
    if isinstance(this_file_directory, bytes):
        sep = b"/"
    else:
        sep = "/"
    tokens = path.split(sep)
    while not re.match(pattern, tokens[-1]):
        print(tokens[-1])
        tokens.pop(-1)

    path = "/".join(tokens)
    return path


# add comfy to path for local devepment only
# find comfy root by going upwards hoping it match regex
# export coffy_local_dev=1
def comfy_root_to_syspath():
    try:
        if os.environ["coffy_local_dev"] == "1":
            print("coffy_local_dev")
        else:
            raise (Exception())
    except:
        return

    path = find_comfy_root()
    if path not in sys.path:
        sys.path.insert(0, path)
