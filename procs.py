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
  from os.path import dirname, basename
  from math import log10
  from importlib import import_module
  MR = import_module(args.mapreduce)

  # Read the header
  from nek import NekFile
  data_path = dirname(args.name)
  data_tag  = basename(args.name) 
  dir_width = int(log10(max(abs(params["io_files"])-1,1)))+1
  if params["io_files"] > 0:
    args.fname = "{:s}{:0{width}d}.f{:05d}".format(args.name, 0, frame, width=dir_width)
  else:
    args.fname = "{:s}/A{:0{width}d}/{:s}{:0{width}d}.f{:05d}".format(data_path, 0, data_tag, 0, frame, width=dir_width)
  print("Opened {:s}".format(args.fname))

  input_file = NekFile(args.fname)

  # Initialize the MapReduce data with base cases
  from copy import deepcopy
  init = MR.MR_init(args, params)
  ans = deepcopy(init)

  # Setup the Map jobs 
  nblock = args.thread
  elm_per_block = int((abs(params["io_files"])*input_file.nelm-1)/args.thread) + 1
  jobs = []
  for j in range(abs(int(params["io_files"]))):
    if params["io_files"] > 0:
      args.fname = "{:s}/{:s}{:0{width}d}.f{:05d}".format(data_path, data_tag, j, frame, width=dir_width)
    else:
      args.fname = "{:s}/A{:0{width}d}/{:s}{:0{width}d}.f{:05d}".format(data_path, j, data_tag, j, frame, width=dir_width)
    input_file = NekFile(args.fname)
    ranges = []
    for i in range(0, input_file.nelm, elm_per_block):
      ranges.append([i, min(i+elm_per_block, input_file.nelm)])
    targs  = zip( ranges,
                  [args.fname] * len(ranges), 
                  [params]     * len(ranges), 
                  [init]       * len(ranges), 
                  [args]       * len(ranges)
	      )
    jobs = jobs + list(targs)

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
  elm_range, fname, params, ans_in, args = job

  # always need this
  import numpy as np
  from importlib import import_module
  MR = import_module(args.mapreduce)

  # Create 'empty' answer dictionary
  from copy import deepcopy
  res = deepcopy(ans_in)
  ans = deepcopy(ans_in)

  # Open the data file
  from nek import NekFile
  input_file = NekFile(fname)
  res['time'] = input_file.time
  print("Processed {:s}".format(fname))

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
