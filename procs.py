"""
Parallel process model: does not require modification
"""


def outer_process(job):  
  """
  Process to be executed in the outer IPython.parallel map
  """

  # always need these
  import numpy as np

  # Split the arguments
  args, params, frame = job

  # Read the header
  from nek import NekFile
  fname = "{:s}0.f{:05d}".format(args.name, frame)
  input_file = NekFile(fname)
  input_file.close()

  # Initialize the MapReduce data with base cases
  from copy import deepcopy
  from MapReduce import MR_init
  init = MR_init(args, params)
  ans = deepcopy(init)

  # Setup the Map jobs 
  nblock = args.thread
  elm_per_block = int(input_file.nelm/args.thread) + 1
  ranges = []
  for i in range(args.thread):
    ranges.append([i*elm_per_block, min((i+1)*elm_per_block, input_file.nelm)])
  targs  = zip( ranges,
                [fname] *nblock, 
		[params]*nblock, 
		[init]   *nblock, 
		[args]  *nblock
	      )
  jobs  = list(targs)

  # Map!
  import time as time_
  ttime = time_.time()
  if args.thread < 1:
    results = map(inner_process, jobs)
  else:
    from multiprocessing import Pool
    p = Pool(processes=args.thread)
    results = p.map(inner_process, jobs, chunksize = 1)
    p.close()
  if args.verbose:
    print('  Map took {:f}s on {:d} processes'.format(time_.time()-ttime, args.thread))

  # Reduce!
  from MapReduce import reduce_
  ttime = time_.time()
  for r in results:
    reduce_(ans, r)
  data = ans['data']
  if args.verbose:
    print('  Reduce took {:f}s on {:d} processes'.format(time_.time()-ttime, args.thread))

  # Analysis! 
  from post import post_frame
  post_frame(ans, args, params, frame, input_file.time)

  return (str(input_file.time), ans)


def inner_process(job):
  """
  Process to be executed  in the inner multiprocessing map
  """
  
  # always need this
  import numpy as np

  # Parse the arguments
  elm_range, fname, params, ans, args = job

  # Create 'empty' answer dictionary
  from copy import deepcopy
  res = deepcopy(ans)

  # Open the data file
  from nek import NekFile
  input_file = NekFile(fname)
  res['time'] = input_file.time

  # Loop over maps and local reduces
  from MapReduce import map_, reduce_
  from tictoc import tic, toc
  for pos in range(elm_range[0], elm_range[1], args.block):
    tic()
    nelm, x, u, p, t = input_file.get_elem(args.block, pos)
    toc('read')

    # All the work is here!
    map_(x, u, p, t, params, ans)

    # This reduce is more of a combiner
    reduce_(res, ans)

  input_file.close()
  return res
