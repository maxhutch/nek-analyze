#!/usr/bin/env python3
"""
Visualize series outputs of nek-analyze
"""

# get arguments
from ui import command_line_ui
args = command_line_ui()

# load params from genrun.py input dictionary
from chest import Chest
import json
#from utils.custom_json import CustomDecoder
with open("{:s}.json".format(args.name), 'r') as f:
  params = json.load(f)

# insert new results into the dictionary
fname = '{:s}-results.dat'.format(args.name)
#with open(fname, 'r') as f:
#  results = json.load(f, cls=CustomDecoder)
results = Chest(path="{:s}-results".format(args.name))

from importlib import import_module
x = import_module(args.post)
for res in results.values():
  x.post_frame(res, params, args)  

# Post-post process the contents of the results dictionary
x.post_series(results, params, args)

