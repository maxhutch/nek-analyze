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
  from importlib import import_module
  MR = import_module(args.mapreduce)

  # Initialize the MapReduce data with base cases
  # Returns job list to pass to map
  jobs = MR.MR_init(args, params, frame)
  # Copy a base case in which to reduce the results
  from copy import deepcopy
  ans = deepcopy(jobs[0][4])

  # Map!
  import time as time_
  ttime = time_.time()
  if args.thread < 2:
    results = map(inner_process, jobs)
  else:
    from multiprocessing import Pool
    p = Pool(processes=args.thread)
    results = p.imap_unordered(inner_process, jobs, chunksize = 1)
  if args.verbose:
    print('  Map took {:f}s on {:d} processes'.format(time_.time()-ttime, args.thread))

  # Reduce!
  ttime = time_.time()
  for r in results:
    MR.reduce_(ans, r)
  if args.thread >= 2:
    p.close()
  if args.verbose:
    print('  Reduce took {:f}s on {:d} processes'.format(time_.time()-ttime, args.thread))

  ans["frame"] = frame

  # Analysis! 
  post = import_module(args.post)
  post.post_frame(ans, params, args)
  post.plot_frame(ans, params, args)

  # Save the results to file!
  from chest import Chest
  cpath = '{:s}-chest-{:03d}'.format(args.name, frame)
  c = Chest(path=cpath)
  for key in ans.keys():
    c[ans['time'], key] = ans[key]
  c.flush()

  return cpath


def inner_process(job):
  """
  Process to be executed  in the inner multiprocessing map
  """
  
  # Parse the arguments
  elm_range, fname, params, args, ans_in = job

  # always need this
  from importlib import import_module
  MR = import_module(args.mapreduce)

  # Create 'empty' answer dictionary
  from copy import deepcopy
  res = ans_in
  #res = deepcopy(ans_in)
  ans = deepcopy(ans_in)

  # Open the data file
  from interfaces.nek.files import NekFile
  input_file = NekFile(fname)
  #res['time'] = input_file.time
  print("Processed {:s}".format(fname))

  # Loop over maps and local reduces
  for pos in range(elm_range[0], elm_range[1], args.block):
    # make sure we don't read past this thread's range
    nelm_to_read = min(args.block, elm_range[1] - pos)

    # All the work is here!
    MR.map_(input_file, pos, nelm_to_read, params, ans)

    # This reduce is more of a combiner
    MR.reduce_(res, ans)

  input_file.close()
  return res

