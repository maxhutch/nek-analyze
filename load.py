#!/usr/bin/env python3
"""
Driver for nek-analyze
"""

# get arguments
from ui import command_line_ui
args = command_line_ui()

# load params from genrun.py input dictionary
import json
with open("{:s}.json".format(args.name), 'r') as f:
  params = json.load(f)

# Set up the frame arguments
from parallel.procs import outer_process
jobs = [[args, params, i] for i in range(args.frame, args.frame_end+1)]

# schedule the frames, one IPython process each
# if only one process or parallel not set, use normal map
import time
start_time = time.time()
if len(jobs) > 2 and args.parallel:
  from IPython.parallel import Client
  p = Client(profile='mpi')
  stuff = p.load_balanced_view().map_async(outer_process, jobs)
else:
  stuff =  map(outer_process, jobs)

# insert new results into the dictionary
from os.path import exists
fname = '{:s}-results.dat'.format(args.name)
from utils.custom_json import CustomEncoder, CustomDecoder
for i, res in enumerate(stuff):

  # load the results dictionary (if it exists)
  results = {}
  if exists(fname):
    with open(fname, 'r') as f:
      results = json.load(f, cls=CustomDecoder)

  # if the frame is already there
  if res[0] in results:
    # merge results
    results[res[0]] = dict(list(results[res[0]].items()) + list(res[1].items()))
  else:
    # insert results
    results[res[0]] = res[1]

  # dump the dictionary back to json file
  with open(fname, 'w') as f:
    json.dump(results,f, indent=2, separators=(',',':'), cls=CustomEncoder)

  # Print a progress update
  run_time = time.time() - start_time
  print("Processed {:d}th frame after {:f}s ({:f} fps)".format(i, run_time, (i+1)/run_time)) 

