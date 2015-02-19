#!/home/maxhutch/anaconda3/bin/python
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
times = set([key[0] for key in results.keys()])
from importlib import import_module
x = import_module(args.post)
for time in times:
  ans = dict([if key[0] == time, (key[1], results[key]) for key in results.keys()])
  x.post_frame(ans, params, args)  

# Post-post process the contents of the results dictionary
x.post_series(results, params, args)

