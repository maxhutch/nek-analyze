def process(job):  

  # Split the arguments
  args = job[0]
  frame = job[1]

  import json
  import numpy as np
  from nek import NekFile
  from tictoc import tic, toc
  from thread_work import tprocess

  from MapReduce import MRInit, Reduce
  from post import post_frame

  from multiprocessing import Pool
  import time as timee

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
  ans = MRInit(args, params)

  # Setup the Map jobs
  nblock = args.thread
  elm_per_block = int(np.product(size)/args.thread) + 1
  ranges = []
  for i in range(args.thread):
    ranges.append([i*elm_per_block, min((i+1)*elm_per_block, np.product(size))])
  targs  = zip( ranges,
                [fname] *nblock, 
		[params]*nblock, 
		[ans]   *nblock, 
		[args]  *nblock
	      )
  jobs  = list(targs)

  # Map!
  ttime = timee.time()
  pool = Pool(args.thread)
  results = pool.map(tprocess, jobs, chunksize = 1)
  print('Map took {:f}s on {:d} processes'.format(timee.time()-ttime, args.thread))

  # Reduce!
  ttime = timee.time()
  for r in results:
    Reduce(ans, r)
  pool.close()
  print('Reduce took {:f}s on {:d} processes'.format(timee.time()-ttime, args.thread))
  data = ans['data']

  # Analysis! 
  post_frame(ans, args, params, frame, input_file.time)

  return (str(input_file.time), ans)

