"""
Parallel process model: does not require modification
"""

def outer_process(job):  
  """
  Process to be executed in the outer IPython.parallel map
  """

  # Split the arguments
  args, params, frame = job

  # always need these
  import numpy as np
  from importlib import import_module
  MR = import_module(args.mapreduce)

  # Read the header
  from nek import NekFile
  args.fname = "{:s}0.f{:05d}".format(args.name, frame)
  input_file = NekFile(args.fname)

  # Initialize the MapReduce data with base cases
  from copy import deepcopy
  init = MR.MR_init(args, params)
  ans = deepcopy(init)

  # Setup the Map jobs 
  nblock = args.thread
  elm_per_block = int((input_file.nelm-1)/args.thread) + 1
  ranges = []
  for i in range(args.thread):
    ranges.append([i*elm_per_block, min((i+1)*elm_per_block, input_file.nelm)])
  targs  = zip( ranges,
                [args.fname] *nblock, 
		[params]*nblock, 
		[init]   *nblock, 
		[args]  *nblock
	      )
  jobs  = list(targs)

  # Map!
  import time as time_
  ttime = time_.time()
  if args.thread < 2:
    results = map(inner_process, jobs)
  else:
    from multiprocessing import Pool
    p = Pool(processes=args.thread)
    results = p.map(inner_process, jobs, chunksize = 1)
    p.close()
  if args.verbose:
    print('  Map took {:f}s on {:d} processes'.format(time_.time()-ttime, args.thread))

  # Reduce!
  ttime = time_.time()
  for r in results:
    MR.reduce_(ans, r)
  if args.verbose:
    print('  Reduce took {:f}s on {:d} processes'.format(time_.time()-ttime, args.thread))

  # Analysis! 
  post = import_module(args.post)
  post.post_frame(ans, args, params, frame, input_file.time)

  return (str(input_file.time), ans)


def inner_process(job):
  """
  Process to be executed  in the inner multiprocessing map
  """
  
  # Parse the arguments
  elm_range, fname, params, ans, args = job

  # always need this
  import numpy as np
  from importlib import import_module
  MR = import_module(args.mapreduce)

  # Create 'empty' answer dictionary
  from copy import deepcopy
  res = deepcopy(ans)

  # Open the data file
  from nek import NekFile
  input_file = NekFile(fname)
  res['time'] = input_file.time

  # Loop over maps and local reduces
  from tictoc import tic, toc
  for pos in range(elm_range[0], elm_range[1], args.block):
    tic()
    # make sure we don't read past this thread's range
    nelm_to_read = min(args.block, elm_range[1] - pos)
    nelm, x, u, p, t = input_file.get_elem(nelm_to_read, pos)
    toc('read')

    # All the work is here!
    MR.map_(x, u, p, t, params, ans)

    # This reduce is more of a combiner
    MR.reduce_(res, ans)

  input_file.close()
  return res
