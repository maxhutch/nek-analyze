#!/usr/bin/env python3
"""
Driver for nek-analyze
"""

# get arguments
from ui import command_line_ui
args = command_line_ui()

# load params from genrun.py input dictionary
import json
with open(args.param_path, 'r') as f:
  params = json.load(f)

# Set up the frame arguments
from mapcombine import outer_process
jobs = [[args, params, i] for i in range(args.frame, args.frame_end+1)]

# schedule the frames, one IPython process each
# if only one process or parallel not set, use normal map
import time
start_time = time.time()
if len(jobs) > 1 and args.parallel:
  from IPython.parallel import Client
  p = Client(profile='mpi')
  stuff = p.load_balanced_view().map_async(outer_process, jobs)
else:
  stuff =  map(outer_process, jobs)

# insert new results into the out-of-core dictionary (Chest)
nelm = params["shape_mesh"][0] * params["shape_mesh"][1] * params["shape_mesh"][2]
from chest import Chest
for i, res in enumerate(stuff):
  c1 = Chest(path=res['cpath'])
  c = Chest(path=args.chest_path)
  c.update(c1)
  c.flush()
  c1.drop()

  # Print a progress update
  run_time = time.time() - start_time
  print("Processed {:d}th frame after {:f}s ({:f} eps)".format(i, run_time, (i+1)*nelm/run_time)) 

