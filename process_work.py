def process(job):  
  # Split the arguments
  args = job[0]
  frame = job[1]

  import gc
  if args.series or not args.display:
    import matplotlib
    matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  import json
  import numpy as np
  from Grid import Grid
  from my_utils import find_root, lagrange_matrix
  from my_utils import transform_field_elements
  from my_utils import transform_position_elements
  from Grid import mixing_zone, energy_budget
  from Grid import plot_slice, plot_spectrum, plot_dist, plot_dim, plot_prof
  from nek import NekFile
  from tictoc import tic, toc
  from thread_work import tprocess

  from MapReduce import merge

  from multiprocessing import Pool
  from multiprocessing import Lock
  import time as timee

  ans = {}
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

  # inits
  ans['PeCell'] = 0.
  ans['ReCell'] = 0.
  ans['TAbs']   = 0.
  ans['TMax']   = 0.
  ans['TMin']   = 0.
  ans['UAbs']   = 0.
  ans['dx_max'] = 0.
  data = Grid(args.ninterp * params['order'], 
              params['root_mesh'], 
              params['extent_mesh'], 
              np.array(params['shape_mesh'], dtype=int) * int(args.ninterp * params['order']),
              boxes = args.boxes)

  # Load file
  fname = "{:s}0.f{:05d}".format(args.name, frame)
  input_file = NekFile(fname)

  nblock = args.thread
  elm_per_block = int(input_file.nelm/args.thread)
  ranges = []
  for i in range(args.thread):
    ranges.append([i*elm_per_block, min((i+1)*elm_per_block, input_file.nelm)])

  targs  = zip( ranges,
                ["Lock()"]*nblock, 
                [fname]*nblock, 
		[params]*nblock, 
		[data]*nblock, 
		[args]*nblock
	      )
  jobs  = list(targs)

  ttime = timee.time()
  pool = Pool(args.thread)
  #with Pool(args.thread) as pool:
  results = pool.map(tprocess, jobs, chunksize = 1)
  for r in results:
    merge(ans, r)
    data.merge(r['data'])
  pool.close()
  print('Thread map took {:f}s on {:d} threads'.format(timee.time()-ttime, args.thread))

  input_file.close()

   
  # more results
  ans['TAbs'] = max(ans['TMax'], -ans['TMin'])
  ans['PeCell'] = ans['UAbs']*ans['dx_max']/params['conductivity']
  ans['ReCell'] = ans['UAbs']*ans['dx_max']/params['viscosity']
  if args.verbose:
    print("Extremal temperatures {:f}, {:f}".format(ans['TMax'], ans['TMin']))
    print("Max speed: {:f}".format(ans['UAbs']))
    print("Cell Pe: {:f}, Cell Re: {:f}".format(ans['PeCell'], ans['ReCell']))
    if args.boxes:
      print("Boxes: " + str(np.log2(data.boxes)))

  center = data.shape[1]/2
  if not args.contour:
    data.cont = None
  else:
    data.cont = np.zeros((data.shape[0], data.shape[1]))
    tic()
    for i in range(data.shape[0]):
      for j in range(data.shape[1]):
        print(i,j)
        data.cont[i,j] = find_root(data.x[i,j,:,2], data.f[i,j,:], desired_resolution = 1.e-15)
    if frame == 1:
      modes_x = np.fft.fftfreq(data.shape[0], data.x[1,0,0,0] - data.x[0,0,0,0]) 
      modes_y = np.fft.rfftfreq(data.shape[1], data.x[0,1,0,1] - data.x[0,0,0,1]) 
      modes = np.zeros((modes_x.size, modes_y.size))
      for i in range(modes_x.size):
        for j in range(modes_y.size):
          modes[i,j] = np.sqrt(abs(modes_x[i])*abs(modes_x[i]) + abs(modes_y[j]) * abs(modes_y[j]))
      np.save("{:s}-cont{:d}".format(args.name, frame), data.cont)
      np.save("{:s}-modes".format(args.name), modes)
    if frame == 2:
      np.save("{:s}-cont{:d}".format(args.name, 2), data.cont)
    toc('contour')

  tic()
  if data.box_dist != None:
    plot_dim(data, fname = "{:s}{:05d}-dim.png".format(args.name, frame)) 

  if args.Fourier:
    plot_spectrum(data, fname = "{:s}{:05d}-spectrum.png".format(args.name, frame), 
                  slices = [.5],
                  contour = args.contour
                 )
  
  # Scatter plot of temperature (slice through pseudocolor in visit)
  if args.slice:
    plot_slice(data, fname = "{:s}{:05d}-zslice.png".format(args.name, frame), zslice=True)
    plot_slice(data, fname = "{:s}{:05d}-yslice.png".format(args.name, frame))

  if args.mixing_cdf:
    plot_dist(data, "{:s}{:05d}-cdf.png".format(args.name, frame))

  toc('plot')

  if args.mixing_zone:
    tic()
    ans['h_cabot'], ans['h_visual'], ans['h_fit'], ans['Xi'], ans['Total'] = mixing_zone(data)
    plot_prof(data, "{:s}{:05d}-prof.png".format(args.name, frame), -1./(2. * ans['h_fit']))
    toc('mixing_zone')

    if not args.series:
      print("Mixing (h_cab,h_vis,h_fit,xi): {:f} {:f} {:f}".format(ans['h_cabot'],ans['h_visual'],ans['h_fit'], ans['Xi']))

  if True:
    tic()
    ans['P'], ans['K'] = energy_budget(data)
    toc('energy_budget')

    if not args.series:
      print("Energy Budget (P,K): {:e} {:e}".format(ans['P'],ans['K']))  

  # free(data)
  data = None; gc.collect()
  
  if not args.series and args.display:
    plt.show()
  plt.close('all')

  return (str(input_file.time), ans)

