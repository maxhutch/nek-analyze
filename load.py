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
  from my_utils import find_root, lagrange_matrix, transform_elements
  from Grid import Grid
  from Grid import mixing_zone
  from Grid import plot_slice
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

  t_trans, pos_trans = transform_elements(t, pos, trans, cart)

  # switch from list of elements to grid
  tic()
  data = Grid(pos_trans, t_trans)
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

  '''
  foo = data.f.ravel()
  print("Extra temperature {:f} \n".format(np.sum(np.abs(np.where(foo > 1, foo, 0)))))
  '''
   
  # Scatter plot of temperature (slice through pseudocolor in visit)
  tic()
  if args.slice:
    plot_slice(data, fname = "{:s}{:05d}-slice.png".format(args.name, frame))

  if args.Fourier:
    # Fourier analysis in 1 dim
    plt.figure()
    if args.contour:
      bx1 = plt.subplot(1,2,1)
    else:
      bx1 = plt.subplot(1,1,1)
    spectrum_center = np.fft.rfft2(data.f[:,:,int(data.shape[2]/2)])
    spectrum_quarter = np.fft.rfft2(data.f[:,:,int(3*data.shape[2]/4)])
    modes_x = np.fft.fftfreq(data.shape[0])
    modes_y = np.fft.rfftfreq(data.shape[1])
    modes = np.zeros((modes_x.size, modes_y.size))
    for i in range(modes_x.size):
      for j in range(modes_y.size):
        modes[i,j] = np.sqrt(abs(modes_x[i])*abs(modes_x[i]) + abs(modes_y[j]) * abs(modes_y[j]))
#    modes = np.sqrt(np.outer(np.arange(int(data.shape[0]/2+1)), np.arange(int(data.shape[1]/2+1))))
    bx1.plot(modes.ravel(), np.abs(spectrum_center.ravel()), 'bo')
    bx1.plot(modes.ravel(), np.abs(spectrum_quarter.ravel()), 'ro')
    plt.title('temperature')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Mode')
    plt.ylabel('Log Amplitude')
#    plt.xlim([0,0.5])
    plt.ylim([10**(-3),10**4])
    if args.contour:
      bx2 = plt.subplot(1,2,2)
      bx2.bar(np.arange(int(data.shape[0]/2+1)),abs(np.fft.rfft(cont)))
      plt.title('contour')
      plt.xlabel('Mode')
      plt.ylabel('Amplitude')
      plt.xlim([0,10])
    plt.savefig("{:s}{:05d}-spectrum.png".format(args.name, frame))
   
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
    plt.show()

