#!/usr/bin/env python3
"""
Visualize series outputs of nek-analyze
"""

# get arguments
from ui import command_line_ui
args = command_line_ui()

# load params from genrun.py input dictionary
import json
with open("{:s}.json".format(args.name), 'r') as f:
  params = json.load(f)

# insert new results into the dictionary
fname = '{:s}-results.dat'.format(args.name)
with open(fname, 'r') as f:
  results = json.load(f)

# Post-post process the contents of the results dictionary
from importlib import import_module
x = import_module(args.post)
x.post_series(results, params, args)

