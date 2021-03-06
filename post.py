"""
Post-processing module: to be completed by user
"""

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

  if args.display:
    plt.show()
    
  return


def post_frame(results, args, params, frame, time):
  """Post-process single frame results, outputting to screen or files.

  Keyword arguments:
  results -- ouput of process_work
  args    -- namespace of commandline arguments from ArgumentParser
  params  -- dictionary of problem parameters read from {name}.json
  frame -- integer frame number
  time -- real time stamp
  """

  if args.series or not args.display:
    import matplotlib
    matplotlib.use('Agg')

  return 

