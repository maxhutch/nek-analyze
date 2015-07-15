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


  return

def post_frame(ans, params, args):
  # Analysis! 
  from numpy.linalg import eigh
  from numpy import argsort

  ev, vecs = eigh(ans["overlap"])
  idx = argsort(ev)[::-1]
  ans["ev"] = ev[idx]
  ans["vecs"] = vecs[:,idx]

  print(ans["overlap"])
  print("Singular values:")
  print(ans["ev"])
  print(ans["vecs"])

  from chest import Chest
  cpath = '{:s}-chest-{:03d}'.format(args.chest_path, ans["frame"])
  c = Chest(path=cpath)
  for key in ans.keys():
    c[ans['time'], key] = ans[key]
  ans.clear()
  c.flush()

  ans["cpath"] = cpath

  return 

