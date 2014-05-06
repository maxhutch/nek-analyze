#!/usr/bin/python3

def process(job):  
  args = job[0]
  frame = job[1]

  import gc
  if args.series or not args.display:
    import matplotlib
    matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  import json
  import numpy as np
  from my_utils import find_root, lagrange_matrix
  from my_utils import transform_field_elements
  from my_utils import TransformFieldElements
  from my_utils import TransformPositionElements
  from Grid import Grid
  from Grid import mixing_zone, energy_budget
  from Grid import plot_slice, plot_spectrum, plot_dist
  from nek import NekFile
  from tictoc import tic, toc
  from threading import Thread

  ans = {}
  # Load params
  with open("{:s}.json".format(args.name), 'r') as f:
    params = json.load(f)

  # inits
  ans['PeCell'] = 0.
  ans['ReCell'] = 0.
  ans['TAbs']   = 0.
  ans['TMax']   = 0.
  ans['TMin']   = 0.
  ans['UAbs']   = 0.

  # Load file
  fname = "{:s}0.f{:05d}".format(args.name, frame)
  input_file = NekFile(fname)
  data = Grid(args.ninterp * params['order'], params['root_mesh'], params['extent_mesh'], np.array(params['shape_mesh'], dtype=int) * int(args.ninterp * params['order']))
  time = input_file.time
  norder = input_file.norder
  block = 32768 #input_file.nelm
  while True:
    tic()
    nelm, pos, vel, t = input_file.get_elem(block)
    toc('read')
    if nelm < 1:
      break

    # Learn about the mesh 
    if True or frame == args.frame:
      origin = np.array(params['root_mesh'])
      corner = np.array(params['extent_mesh'])
      extent = corner-origin
      size = np.array(params['shape_mesh'], dtype=int)
      if args.verbose:
        print("Grid is ({:f}, {:f}, {:f}) [{:d}x{:d}x{:d}] with order {:d}".format(
              extent[0], extent[1], extent[2], size[0], size[1], size[2], norder))

      # setup the transformation
      ninterp = int(args.ninterp*norder)
      gll  = pos[0:norder,0,0] - pos[0,0,0]
      dx_max = np.max(gll[1:] - gll[0:-1])
      cart = np.linspace(0.,extent[0],num=ninterp,endpoint=False)/size[0]
      trans = lagrange_matrix(gll,cart)
      if args.verbose:
        print("Interpolating\n" + str(gll) + "\nto\n" + str(cart))

    x_thread = TransformPositionElements(pos, trans, cart)
    x_thread.start()
    x_thread.join()
    pos_trans = x_thread.p_trans
    del x_thread; pos = None; gc.collect()

    t_thread = TransformFieldElements(t, trans, cart)
    t_thread.start()
    t_thread.join()
    t_trans = t_thread.f_trans 
    del t_thread; t = None; gc.collect()

    uz_thread = TransformFieldElements(vel[:,:,2], trans, cart)
    uz_thread.start()
    uz_thread.join()
    uz_trans = uz_thread.f_trans 
    del uz_thread; vel = np.delete(vel, 2, 2); gc.collect()

    uy_thread = TransformFieldElements(vel[:,:,1], trans, cart)
    uy_thread.start()
    uy_thread.join()
    uy_trans = uy_thread.f_trans 
    del uy_thread; vel = np.delete(vel, 1, 2); gc.collect()

    ux_thread = TransformFieldElements(vel[:,:,0], trans, cart)
    ux_thread.start()
    ux_thread.join()
    ux_trans = ux_thread.f_trans 
    del ux_thread; vel = None; gc.collect()

    # Print some stuff 
    max_speed = np.sqrt(np.max(np.square(ux_trans) + np.square(uy_trans) + np.square(uz_trans)))
    ans['TMax']   = float(max(ans['TMax'], np.amax(t_trans)))
    ans['TMin']   = float(min(ans['TMin'], np.amin(t_trans)))
    ans['UAbs']   = float(max(ans['UAbs'], max_speed))

    # Renorm
    tic()
    Tt_low = -0.0005; Tt_high = 0.0005
    t_trans = (t_trans - Tt_low)/(Tt_high - Tt_low)
    t_trans = np.maximum(t_trans, 0.)
    t_trans = np.minimum(t_trans, 1.)
    toc('renorm')

    # switch from list of elements to grid
    data.add(pos_trans, f_elm = t_trans)
    t_trans = None; gc.collect()
    data.add(pos_trans, ux_elm = ux_trans)
    ux_trans = None; gc.collect()
    data.add(pos_trans, uy_elm = uy_trans)
    uy_trans = None; gc.collect()
    data.add(pos_trans, uz_elm = uz_trans)
    uz_trans = None; gc.collect()
    pos_trans = None; gc.collect() 

  input_file.close()
  data.add_pos()

  ans['TAbs'] = max(ans['TMax'], -ans['TMin'])
  ans['PeCell'] = ans['UAbs']*dx_max/params['conductivity']
  ans['ReCell'] = ans['UAbs']*dx_max/params['viscosity']
  if args.verbose:
    print("Extremal temperatures {:f}, {:f}".format(ans['TMax'], ans['TMin']))
    print("Max speed: {:f}".format(ans['UAbs']))
    print("Cell Pe: {:f}, Cell Re: {:f}".format(ans['PeCell'], ans['ReCell']))

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

  # Scatter plot of temperature (slice through pseudocolor in visit)
  data.f = None; data.ux = None; data.uy = None; data.uz = None

  tic()
  if args.Fourier:
    plot_spectrum(data, fname = "{:s}{:05d}-spectrum.png".format(args.name, frame), 
                  slices = [.5],
                  contour = args.contour
                 )
  
  if args.slice:
    plot_slice(data, fname = "{:s}{:05d}-slice.png".format(args.name, frame))

  if args.mixing_cdf:
    plot_dist(data, "{:s}{:05d}-cdf.png".format(args.name, frame))

  toc('plot')

  if args.mixing_zone:
    tic()
    h_cabot, h_visual, X = mixing_zone(data)
    ans['h_cabot'] = h_cabot
    ans['h_visual'] = h_visual
    ans['Xi'] = X
    toc('mixing_zone')

    if not args.series:
      print("Mixing (h_cab,h_vis,xi): {:f} {:f} {:f}".format(h_cabot,h_visual,X))

  if True:
    tic()
    P, K = energy_budget(data)
    ans['P'] = P
    ans['K'] = K
    toc('energy_budget')

    if not args.series:
      print("Energy Budget (P,K): {:e} {:e}".format(P,K))  

  data = None; gc.collect()
  
  if not args.series and args.display:
    plt.show()
  plt.close('all')

  return (str(time), ans)
#========================================================================================

import numpy as np
import json
from os.path import exists
from tictoc import tic, toc
import time

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("name", help="Nek *.fld output file")
parser.add_argument("-f", "--frame", help="[Starting] Frame number", type=int, default=1)
parser.add_argument("-e", "--frame_end", help="Ending frame number", type=int, default=-1)
parser.add_argument("-s", "--slice", help="Display slice", action="store_true")
parser.add_argument("-c", "--contour", help="Display contour", action="store_true")
parser.add_argument("-n", "--ninterp", help="Interpolating order", type=float, default = 1.)
parser.add_argument("-z", "--mixing_zone", help="Compute mixing zone width", action="store_true")
parser.add_argument("-m", "--mixing_cdf", help="Plot CDF of box temps", action="store_true")
parser.add_argument("-F", "--Fourier", help="Plot Fourier spectrum in x-y", action="store_true")
parser.add_argument("-d", "--display", help="Display plots with X", action="store_true", default=False)
parser.add_argument("-v", "--verbose", help="Should I be really verbose, that is wordy?", action="store_true", default=False)
args = parser.parse_args()
if args.frame_end == -1:
  args.frame_end = args.frame
args.series = (args.frame != args.frame_end)

if not args.display:
  import matplotlib
  matplotlib.use('Agg')
import matplotlib.pyplot as plt


""" Load the data """
from toolz.curried import map

# Load params
with open("{:s}.json".format(args.name), 'r') as f:
  params = json.load(f)
params['g'] = 9.8

jobs = [[args, i] for i in range(args.frame, args.frame_end+1)]
start_time = time.time()
if len(jobs) > 2:
  from IPython.parallel import Client
  p = Client(profile='default')
  pmap = p.load_balanced_view().map_async
  stuff = pmap(process, jobs)
else:
  stuff = map(process, jobs)

fname = '{:s}-results.dat'.format(args.name)
results = {}
if exists(fname):
  with open(fname, 'r') as f:
    results = json.load(f)

for i, res in enumerate(stuff):
  if res[0] in results:
    results[res[0]] = dict(list(results[res[0]].items()) + list(res[1].items()))
  else:
    results[res[0]] = res[1]
  with open(fname, 'w') as f:
    json.dump(results,f)
  run_time = time.time() - start_time
  print("Processed {:d}th frame after {:f}s ({:f} fps)".format(i, run_time, i/run_time)) 

from os import system
if args.series: 
  results_with_times = sorted([[float(elm[0]), elm[1]] for elm in results.items()])
  times, vals = zip(*results_with_times)


  PeCs  = [d['PeCell'] for d in vals]
  TMaxs = np.array([d['TAbs'] for d in vals])
  plt.figure()
  ax1 = plt.subplot(1,1,1)
  plt.xlabel('Time (s)')
  ax1.plot(times, PeCs, label='Cell Peclet')
  ax1.plot(times, TMaxs*2./params['atwood'], label='max(T)/max(T0)')
  plt.legend(loc=2)
  plt.savefig("{:s}-stability.png".format(args.name))

  if args.slice:
    system("rm -f "+args.name+"-slice.mkv")
    system("avconv -f image2 -i "+args.name+"%05d-slice.png -c:v h264 "+args.name+"-slice.mkv")
  if args.mixing_cdf:
    system("rm -f "+args.name+"-cdf.mkv")
    system("avconv -f image2 -i "+args.name+"%05d-cdf.png -c:v h264 "+args.name+"-cdf.mkv")
  if args.Fourier:
    system("rm -f "+args.name+"-spectrum.mkv")
    system("avconv -f image2 -i "+args.name+"%05d-spectrum.png -c:v h264 "+args.name+"-spectrum.mkv") 
  if args.mixing_zone: 
    hs_cabot = [d['h_cabot'] for d in vals]
    vs_cabot = [(hs_cabot[i+1] - hs_cabot[i-1])/(float(times[i+1])-float(times[i-1])) for i in range(1,len(hs_cabot)-1)]
    vs_cabot.insert(0,0.); vs_cabot.append(0.)
    alpha_cabot = [vs_cabot[i]*vs_cabot[i]/(4*params['atwood']*params['g']*hs_cabot[i]) for i in range(len(vs_cabot))]

    hs_visual = [d['h_visual'] for d in vals]
    vs_visual = [(hs_visual[i+1] - hs_visual[i-1])/(float(times[i+1])-float(times[i-1])) for i in range(1,len(hs_visual)-1)]
    vs_visual.insert(0,0.); vs_visual.append(0.)
    alpha_visual = [vs_visual[i]*vs_visual[i]/(4*params['atwood']*params['g']*hs_visual[i]) for i in range(len(vs_visual))]

    plt.figure()
    ax1 = plt.subplot(1,3,1)
    plt.xlabel('Time (s)')
    plt.ylabel('h (m)')
    plt.ylim([0., max(hs_visual)])
    ax1.plot(times, hs_cabot, times, hs_visual)

    ax2 = plt.subplot(1,3,2)
    plt.ylim([0., max(alpha_visual)])
    ax2.plot(times, alpha_cabot, label='Cabot')
    ax2.plot(times, alpha_visual, label='Visual')
    plt.legend(loc=2)
    plt.xlabel('Time (s)')
    plt.ylabel('alpha')

    Xs = [d['Xi'] for d in vals]
    ax3 = plt.subplot(1,3,3)
    plt.ylim([0.,1.])
    ax3.plot(times, Xs)
    plt.xlabel('Time (s)')
    plt.ylabel('Xi')
    plt.savefig("{:s}-h.png".format(args.name))

    plt.figure()
    Ps = [d['P'] for d in vals]
    Ks = [d['K'] for d in vals]
    ax1 = plt.subplot(1,1,1)
    ax1.plot(times, np.divide(Ps, np.square(hs_cabot)), label='Potential')
    ax1.plot(times, np.divide(Ks, np.square(hs_cabot)), label='Kinetic')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy / h^2')
    plt.legend(loc=2)
    plt.savefig("{:s}-energy.png".format(args.name))

if  args.contour:
    cont1 = np.load("{:s}-cont{:d}.npy".format(args.name, 1))
    cont2 = np.load("{:s}-cont{:d}.npy".format(args.name, 2))
    modes = np.load("{:s}-modes.npy".format(args.name))
    kont1 = np.fft.rfft2(cont1)/cont1.size
    kont2 = np.fft.rfft2(cont2)/cont1.size
    plt.figure()
    ax1 = plt.subplot(1,1,1)
    ax1.plot(modes.ravel(), np.abs(kont1).ravel(), 'o')
    ax1.plot(modes.ravel(), np.abs(kont2).ravel(), 'o')
    plt.figure()
    ax1 = plt.subplot(1,1,1)
    ax1.plot(modes.ravel(), np.log(np.divide(np.abs(kont2), np.abs(kont1))).ravel()/0.01, 'bo') 
#    plt.xscale('log')
#    plt.yscale('log')

if args.display:
  plt.show()
    
