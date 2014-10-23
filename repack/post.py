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

  # Nothing to see here
    
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

  # Close the input and output files and delete their references
  results["output_file"].close()
  del results["output_file"]
  del results["input_file"]

  return 

