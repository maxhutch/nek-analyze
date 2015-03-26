#!/home/maxhutch/anaconda3/bin/python
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
from chest import Chest
c = Chest(path='{:s}-results'.format(args.name))
for i, res in enumerate(stuff):
  c1 = Chest(path=res['cpath'])
  c.update(c1)
  c1.drop()

  # Print a progress update
  run_time = time.time() - start_time
  print("Processed {:d}th frame after {:f}s ({:f} fps)".format(i, run_time, (i+1)/run_time)) 
c.flush()

