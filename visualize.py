#!/usr/bin/env python3
"""
Visualize series outputs of nek-analyze
"""

# get arguments
from ui import command_line_ui
args = command_line_ui()

# load params from genrun.py input dictionary
import json
#from utils.custom_json import CustomDecoder
with open("{:s}.json".format(args.name), 'r') as f:
  params = json.load(f)

# insert new results into the dictionary
fname = '{:s}-results.dat'.format(args.name)
#with open(fname, 'r') as f:
#  results = json.load(f, cls=CustomDecoder)
from chest import Chest
from slict import CachedSlict
results = CachedSlict(Chest(path="{:s}-results".format(args.name)))

from importlib import import_module
xx = import_module(args.post)
import time as clock
start_time = clock.time()
i = 0
#for time in results[:,"frame"].keys():
#  xx.plot_frame(results[time,:], params, args)  
#  i = i + 1
#  print("Processed t={:f} ({:f} fps)".format(time, (clock.time() - start_time) / i))

# Post-post process the contents of the results dictionary
xx.post_series(results, params, args)

