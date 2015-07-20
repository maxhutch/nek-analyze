"""
Post-processing module: to be completed by user
"""

from utils.struct import Struct

def post_series(results, params, args):
  """Post-process time-series results, outputting to screen or files.

  Keyword arguments:
  results -- dictionary of ouputs of process_work keyed by time
  params  -- dictionary of problem parameters read from {name}.json
  args    -- namespace of commandline arguments from ArgumentParser
  """

  import numpy as np
  import matplotlib
  if not args.display:
    matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  # Make namespace-like handles for params
  p = Struct(params)

  return

def plot_frame(ans, params, args):
  # Analysis! 

  return 

def post_frame(ans, params, args):

  from chest import Chest
  cpath = '{:s}-chest-{:03d}'.format(args.chest_path, ans["frame"])
  c = Chest(path=cpath)
  for key in ans.keys():
    c[ans['time'], key] = ans[key]
  ans.clear()
  c.flush()

  ans["cpath"] = cpath

  return 

