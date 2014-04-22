#!/usr/bin/python3

import numpy as np
from sys import argv
import matplotlib.pyplot as plt
import scipy as sp
import scipy.linalg
from scipy.stats import linregress

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

""" Load the data """
quiet = not args.verbose

from sys import path
path.append('./python/')
from nek import from_nek
from my_utils import find_root, lagrange_matrix, transform_elements
from tictoc import *
from Grid import Grid
from Grid import mixing_zone
from Grid import plot_slice
from Grid import fractal_dimension

if args.frame_end == -1:
  args.frame_end = args.frame

for frame in range(args.frame, args.frame_end+1):
  # Load file
  tic()
  fname = "{:s}0.f{:05d}".format(args.name, frame)
  pos, vel, t, speed, time, norder = from_nek(fname)
  nelm = t.shape[1]
  toc('read')

  # Print some stuff 
  if not quiet:
    print("Extremal temperatures {:f}, {:f}".format(np.max(t), np.min(t)))
    print("Extremal u_z          {:f}, {:f}".format(np.max(vel[:,:,2]), np.min(vel[:,:,2])))
    print("Max speed: {:f}".format(np.max(speed)))

  # Learn about the mesh 
  if frame == args.frame:
    origin = np.array([np.min(pos[:,:,0].flatten()),
                       np.min(pos[:,:,1].flatten()), 
                       np.min(pos[:,:,2].flatten())])
    corner = np.array([np.max(pos[:,:,0].flatten()), 
                       np.max(pos[:,:,1].flatten()), 
                       np.max(pos[:,:,2].flatten())])
    extent = corner-origin
    size = np.array((corner - origin)/(pos[0,1,0] - pos[0,0,0]), dtype=int)
    if not quiet:
      print("Grid is ({:f}, {:f}, {:f}) [{:d}x{:d}x{:d}] with order {:d}".format(
            extent[0], extent[1], extent[2], size[0], size[1], size[2], norder))

    # setup the transformation
    ninterp = int(args.ninterp*norder)
    gll  = pos[0:norder,0,0] - pos[0,0,0]
    dx_max = np.max(gll[1:] - gll[0:-1])
    cart = np.linspace(0.,extent[0],num=ninterp,endpoint=False)/size[0]
    trans = lagrange_matrix(gll,cart)
    if not quiet:
      print("Interpolating\n" + str(gll) + "\nto\n" + str(cart))
      
  if not quiet:
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
    boundary, X = mixing_zone(data, plot = False, fname = 'mixing_zone_2.dat', time = time)
    print("Mixing function: {:f}".format(X))

  '''
  foo = data.f.ravel()
  print("Extra temperature {:f} \n".format(np.sum(np.abs(np.where(foo > 1, foo, 0)))))
  '''
   
  # Scatter plot of temperature (slice through pseudocolor in visit)
  tic()
  if args.slice:
    plot_slice(data, contour = cont, fname = "{:s}{:05d}-slice.png".format(args.name, frame))

  if args.Fourier:
    # Fourier analysis in 1 dim
    plt.figure()
    bx1 = plt.subplot(1,2,1)
    bx1.bar(  np.arange(int(data.shape[0]/2+1)),  abs(np.fft.rfft(data.f[:,center,int(data.shape[2]/2)])))
    plt.title('temperature')
    plt.xlabel('Mode')
    plt.ylabel('Amplitude')
    plt.xlim([0,10])
    bx2 = plt.subplot(1,2,2)
    if args.contour:
      bx2.bar(np.arange(int(data.shape[0]/2+1)),abs(np.fft.rfft(cont)))
    plt.title('contour')
    plt.xlabel('Mode')
    plt.ylabel('Amplitude')
    plt.xlim([0,10])
    plt.savefig(argv[1]+'_spectrum.png')
   
  if args.mixing_cdf:
    plt.figure()
    ax1 = plt.subplot(1,1,1)
    ax1.hist(data.f.flatten(), bins=1000, normed=True, range=(-0.1,1.1), cumulative=True)
    plt.xlim([-.1,1.1])
    plt.ylim([0,1])
    plt.savefig("{:s}{:05d}-cdf.png".format(args.name, frame))
  
  if args.frame == args.frame_end:
    plt.show()
  plt.close('all')
  toc('plot')

from os import system
if args.frame != args.frame_end:
  if args.slice:
    system("rm -f "+args.name+"-slice.mkv")
    system("avconv -f image2 -i "+args.name+"%05d-slice.png -c:v h264 "+args.name+"-slice.mkv")
  if args.mixing_cdf:
    system("rm -f "+args.name+"-cdf.mkv")
    system("avconv -f image2 -i "+args.name+"%05d-cdf.png -c:v h264 "+args.name+"-cdf.mkv")


