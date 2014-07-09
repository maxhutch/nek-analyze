#!/usr/bin/env python3
import numpy as np
import json
from os.path import exists
from process_work import process
from post import post_series
import time

# Define arguments
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("name",                 help="Nek *.fld output file")
parser.add_argument("-f",  "--frame",       help="[Starting] Frame number", type=int, default=1)
parser.add_argument("-e",  "--frame_end",   help="Ending frame number", type=int, default=-1)
parser.add_argument("-s",  "--slice",       help="Display slice", action="store_true")
parser.add_argument("-c",  "--contour",     help="Display contour", action="store_true")
parser.add_argument("-n",  "--ninterp",     help="Interpolating order", type=float, default = 1.)
parser.add_argument("-z",  "--mixing_zone", help="Compute mixing zone width", action="store_true")
parser.add_argument("-m",  "--mixing_cdf",  help="Plot CDF of box temps", action="store_true")
parser.add_argument("-F",  "--Fourier",     help="Plot Fourier spectrum in x-y", action="store_true")
parser.add_argument("-b",  "--boxes",       help="Compute box covering numbers", action="store_true")
parser.add_argument("-nb", "--block",       help="Number of elements to process at a time", type=int, default=65536)
parser.add_argument("-nt", "--thread",      help="Number of threads to spawn", type=int, default=1)
parser.add_argument("-d",  "--display",     help="Display plots with X", action="store_true", default=False)
parser.add_argument("-p",  "--parallel",    help="Use parallel map (IPython)", action="store_true", default=False)
parser.add_argument(       "--series",      help="Apply time-series analyses", action="store_true", default=False)
parser.add_argument("-v",  "--verbose",     help="Should I be really verbose, that is: wordy?", action="store_true", default=False)

# Load the arguments
args = parser.parse_args()
if args.frame_end == -1:
  args.frame_end = args.frame
args.series = (args.frame != args.frame_end) or args.series

# Load params
with open("{:s}.json".format(args.name), 'r') as f:
  params = json.load(f)

# Schedule the frames
jobs = [[args, i] for i in range(args.frame, args.frame_end+1)]
start_time = time.time()
if len(jobs) > 2 and args.parallel:
  from IPython.parallel import Client
  p = Client(profile='default')
  pmap = p.load_balanced_view().map_async
  stuff = pmap(process, jobs)
else:
  stuff =  map(process, jobs)

# Insert new results into the dictionary
fname = '{:s}-results.dat'.format(args.name)
for i, res in enumerate(stuff):

  # load the results dictionary
  results = {}
  if exists(fname):
    with open(fname, 'r') as f:
      results = json.load(f)

  # if the frame is already there
  if res[0] in results:
    # merge results
    results[res[0]] = dict(list(results[res[0]].items()) + list(res[1].items()))
  else:
    # insert results
    results[res[0]] = res[1]

  # dump the dictionary
  with open(fname, 'w') as f:
    json.dump(results,f, indent=2, separators=(',',':'))

  # Print a progress update
  run_time = time.time() - start_time
  print("Processed {:d}th frame after {:f}s ({:f} fps)".format(i, run_time, (i+1)/run_time)) 

# Post-post process the contents of the results dictionary
post_series(results, params, args)

