#!/usr/bin/python3

import numpy as np
from sys import argv
import matplotlib.pyplot as plt
import scipy as sp
import scipy.linalg
import json
from scipy.stats import linregress
from os.path import exists

from sys import path
path.append('./python/')
from Grid import Grid
from Grid import fractal_dimension
from tictoc import tic, toc


def process(job):  
  args = job[0]
  frame = job[1]

  import matplotlib.pyplot as plt
  from os.path import exists
  import numpy as np
  from sys import path
  path.append('./python/')
  from my_utils import find_root, lagrange_matrix
  from my_utils import transform_field_elements
  from my_utils import transform_position_elements
  from Grid import Grid
  from Grid import mixing_zone, energy_budget
  from Grid import plot_slice, plot_spectrum
  from Grid import fractal_dimension
  from nek import from_nek
  from tictoc import tic, toc

  ans = {}
  # Load file
  tic()
  fname = "{:s}0.f{:05d}".format(args.name, frame)
  pos, vel, t, speed, time, norder = from_nek(fname)
  nelm = t.shape[1]
  toc('read')

  # Print some stuff 
  if args.verbose:
    print("Extremal temperatures {:f}, {:f}".format(np.max(t), np.min(t)))
    print("Extremal u_z          {:f}, {:f}".format(np.max(vel[:,:,2]), np.min(vel[:,:,2])))
    print("Max speed: {:f}".format(np.max(speed)))

  # Learn about the mesh 
  if True or frame == args.frame:
    origin = np.array([np.min(pos[:,:,0].flatten()),
                       np.min(pos[:,:,1].flatten()), 
                       np.min(pos[:,:,2].flatten())])
    corner = np.array([np.max(pos[:,:,0].flatten()), 
                       np.max(pos[:,:,1].flatten()), 
                       np.max(pos[:,:,2].flatten())])
    extent = corner-origin
    size = np.array((corner - origin)/(pos[0,1,0] - pos[0,0,0]), dtype=int)
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
      
  if args.verbose:
    print("Cell Pe: {:f}, Cell Re: {:f}".format(np.max(speed)*dx_max/2.e-9, np.max(speed)*dx_max/8.9e-7))

  pos_trans = transform_position_elements(pos, trans, cart)
  t_trans = transform_field_elements(t, trans, cart)
  ux_trans = transform_field_elements(vel[:,:,0], trans, cart)
  uy_trans = transform_field_elements(vel[:,:,1], trans, cart)
  uz_trans = transform_field_elements(vel[:,:,2], trans, cart)
#  speed_trans = transform_field_elements(speed, trans, cart)

  # switch from list of elements to grid
  tic()
  data = Grid(pos_trans, t_trans, ux_trans, uy_trans, uz_trans)
  toc('to_grid')

  #print(fractal_dimension(data))

  # Renorm
  Tt_low = -0.0005; Tt_high = 0.0005
  data.f = (data.f - Tt_low)/(Tt_high - Tt_low)
  data.f = np.maximum(data.f, 0.)
  data.f = np.minimum(data.f, 1.)

  center = data.shape[1]/2
  if not args.contour:
    cont = None
  else:
    cont = np.zeros((data.shape[0]))
    tic()
    for i in range(data.shape[0]):
     cont[i] = find_root(data.x[i,center,:,2], data.f[i,center,:])
    toc('contour')

  if args.mixing_zone:
    h_cabot, h_visual, X = mixing_zone(data)
    ans['h_cabot'] = h_cabot
    ans['h_visual'] = h_visual
    ans['Xi'] = X

    if not args.series:
      print("Mixing (h_cab,h_vis,xi): {:f} {:f} {:f}".format(h_cabot,h_visual,X))

  if True:
    P, K = energy_budget(data)
    ans['P'] = P
    ans['K'] = K

    if not args.series:
      print("Energy Budget (P,K): {:e} {:e}".format(P,K))

  '''
  foo = data.f.ravel()
  print("Extra temperature {:f} \n".format(np.sum(np.abs(np.where(foo > 1, foo, 0)))))
  '''
   
  # Scatter plot of temperature (slice through pseudocolor in visit)
  tic()
  if args.slice:
    plot_slice(data, fname = "{:s}{:05d}-slice.png".format(args.name, frame))

  if args.Fourier:
    plot_spectrum(data, fname = "{:s}{:05d}-spectrum.png".format(args.name, frame), slices = [.5])
  
  if args.mixing_cdf:
    plt.figure()
    ax1 = plt.subplot(1,1,1)
    ax1.hist(data.f.flatten(), bins=1000, normed=True, range=(-0.1,1.1), cumulative=True)
    plt.xlim([-.1,1.1])
    plt.ylim([0,1])
    plt.savefig("{:s}{:05d}-cdf.png".format(args.name, frame))

  if not args.series:
    plt.show()
  plt.close('all')
  toc('plot')
  return (str(time), ans)

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
parser.add_argument("-v", "--verbose", help="Should I be really verbose, that is wordy?", action="store_true", default=False)
args = parser.parse_args()
if args.frame_end == -1:
  args.frame_end = args.frame
args.series = (args.frame != args.frame_end)

""" Load the data """
from toolz.curried import map
from IPython.parallel import Client
p = Client(profile='default')[:]
pmap = p.map_sync
jobs = [[args, i] for i in range(args.frame, args.frame_end+1)]
if len(jobs) > 2:
  stuff = pmap(process, jobs)
else:
  stuff = map(process, jobs)

for job in stuff:
  continue

from os import system
Atwood = 1.e-3
g = 9.8
if args.series:
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
    fname = './{:s}-mixing.dat'.format(args.name)
    mixing_dict = {}
    if exists(fname):
      with open(fname, 'r') as f:
        mixing_dict = json.load(f)
   
    for res in stuff:
      if res[0] in mixing_dict:
        mixing_dict[res[0]] = dict(list(mixing_dict[res[0]].items()) + list(res[1].items()))
      else:
        mixing_dict[res[0]] = res[1]
    with open(fname, 'w') as f:
      json.dump(mixing_dict,f)
    
    mixing_dict2 = [[float(elm[0]), elm[1]] for elm in mixing_dict.items()]
    time_series = sorted(mixing_dict2)
    times, vals = zip(*time_series)
    hs_cabot = [d['h_cabot'] for d in vals]
    hs_visual = [d['h_visual'] for d in vals]
    Xs = [d['Xi'] for d in vals]
    Ps = [d['P'] for d in vals]
    Ks = [d['K'] for d in vals]

    vs    = [(hs_cabot[i+1] - hs_cabot[i-1])/(float(times[i+1])-float(times[i-1])) for i in range(1,len(hs_cabot)-1)]
    vs.insert(0,0.); vs.append(0.)
    alpha_cabot = [vs[i]*vs[i]/(4*Atwood*g*hs_cabot[i]) for i in range(len(vs))]

    vs    = [(hs_visual[i+1] - hs_visual[i-1])/(float(times[i+1])-float(times[i-1])) for i in range(1,len(hs_visual)-1)]
    vs.insert(0,0.); vs.append(0.)
    alpha_visual = [vs[i]*vs[i]/(4*Atwood*g*hs_visual[i]) for i in range(len(vs))]

    plt.figure()
    ax1 = plt.subplot(1,3,1)
    plt.ylim([0., max(hs_visual)])
    ax1.plot(times, hs_cabot, times, hs_visual)

    ax2 = plt.subplot(1,3,2)
    plt.ylim([0., max(alpha_visual)])
    ax2.plot(times, alpha_cabot,times, alpha_visual)

    ax3 = plt.subplot(1,3,3)
    plt.ylim([0.,1.])
    ax3.plot(times, Xs)

    plt.figure()
    ax1 = plt.subplot(1,1,1)
    ax1.plot(times, np.divide(Ps, np.square(hs_cabot)), times, np.divide(Ks, np.square(hs_cabot)))
    plt.show()

