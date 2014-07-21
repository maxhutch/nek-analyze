"""
Parallel process model: does not require modification
"""

def outer_process(job):  
  """
  Process to be executed in the outer IPython.parallel map
  """

  # Split the arguments
  args = job[0]
  frame = job[1]

  import json
  import numpy as np
  from nek import NekFile
  from tictoc import tic, toc

  from MapReduce import MRInit, Reduce
  from post import post_frame

  from multiprocessing import Pool
  import time as timee
  from copy import deepcopy


  # Load params
  with open("{:s}.json".format(args.name), 'r') as f:
    params = json.load(f)
  extent = np.array(params['extent_mesh']) - np.array(params['root_mesh'])
  size = np.array(params['shape_mesh'], dtype=int)
  ninterp = int(args.ninterp*params['order'])
  cart = np.linspace(0.,extent[0],num=ninterp,endpoint=False)/size[0]
  if args.verbose:
    print("Grid is ({:f}, {:f}, {:f}) [{:d}x{:d}x{:d}] with order {:d}".format(
          extent[0], extent[1], extent[2], size[0], size[1], size[2], params['order']))
  trans = None

  # Load file header
  fname = "{:s}0.f{:05d}".format(args.name, frame)
  input_file = NekFile(fname)
  input_file.close()

  # Initialize the MapReduce data
  init = MRInit(args, params)
  ans = deepcopy(init)

  # Setup the Map jobs
  nblock = args.thread
  elm_per_block = int(np.product(size)/args.thread) + 1
  ranges = []
  for i in range(args.thread):
    ranges.append([i*elm_per_block, min((i+1)*elm_per_block, np.product(size))])
  targs  = zip( ranges,
                [fname] *nblock, 
		[params]*nblock, 
		[init]   *nblock, 
		[args]  *nblock
	      )
  jobs  = list(targs)

  # Map!
  ttime = timee.time()
  if args.thread > 1:
    pool = Pool(args.thread)
    results = pool.map(inner_process, jobs, chunksize = 1)
  else: 
    results = [inner_process(jobs[0])]
  print('Map took {:f}s on {:d} processes'.format(timee.time()-ttime, args.thread))

  # Reduce!
  ttime = timee.time()
  for r in results:
    Reduce(ans, r)
  if args.thread > 1:
    pool.close()
  print('Reduce took {:f}s on {:d} processes'.format(timee.time()-ttime, args.thread))
  data = ans['data']

  # Analysis! 
  post_frame(ans, args, params, frame, input_file.time)

  return (str(input_file.time), ans)

def inner_process(job):
  """
  Process to be executed  in the inner multiprocessing map
  """

  # Parse the arguments
  elm_range, fname, params, ans, args = job

  import numpy as np
  from nek import NekFile
  from tictoc import tic, toc
  from MapReduce import Map, Reduce
  from copy import deepcopy

  # Create 'empty' answer dictionary
  res = deepcopy(ans)

  input_file = NekFile(fname)

  for pos in range(elm_range[0], elm_range[1], args.block):
    tic()
    nelm_to_read = min(args.block, elm_range[1] - pos)
    nelm, x, u, p, t = input_file.get_elem(nelm_to_read, pos)
    toc('read')

    if nelm < 1:
      input_file.close()
      return res

    Map(x, u, p, t, params, ans)

    # This reduce is more of a combiner
    Reduce(res, ans)

  input_file.close()
  return res
