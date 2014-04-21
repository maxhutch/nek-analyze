#!/usr/bin/python3

import numpy as np
from sys import argv
import matplotlib.pyplot as plt
import scipy as sp
import scipy.linalg
from scipy.stats import linregress

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("fname", help="Nek *.fld output file")
parser.add_argument("-f", "--frame", help="[Starting] Frame number", type=int, default=1)
parser.add_argument("-e", "--frame_end", help="Ending frame number", type=int, default=-1)
parser.add_argument("-s", "--slice", help="Display slice", action="store_true")
parser.add_argument("-c", "--contour", help="Display contour", action="store_true")
parser.add_argument("-n", "--ninterp", help="Interpolating order", type=float, default = 1.)
parser.add_argument("-z", "--mixing_zone", help="Compute mixing zone width", action="store_true")
parser.add_argument("-m", "--mixing_cdf", help="Plot CDF of box temps", action="store_true")
parser.add_argument("-F", "--Fourier", help="Plot Fourier spectrum in x-y", action="store_true")
args = parser.parse_args()

""" Load the data """
quiet = False

from sys import path
path.append('./python/')
from nek import from_nek
from my_utils import find_root, lagrange_matrix
from tictoc import *
from Grid import Grid
from Grid import mixing_zone
from Grid import plot_slice
from Grid import fractal_dimension

if args.frame_end = -1:
  args.frame_end = args.frame

tic()
fname = "{:s}0.f{:05d}".format(args.fname, args.frame)
pos, vel, t, speed, time, norder = from_nek(fname)
nelm = t.shape[1]
toc('read')

#''' Print some stuff '''
if not quiet:
  print("Extremal temperatures {:f}, {:f}".format(np.max(t), np.min(t)))
  print("Extremal u_z          {:f}, {:f}".format(np.max(vel[:,:,2]), np.min(vel[:,:,2])))
  print("Max speed: {:f}".format(np.max(speed)))

#''' Learn about the mesh '''
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
  print("Cell Pe: {:f}, Cell Re: {:f}".format(np.max(speed)*dx_max/2.e-9, np.max(speed)*dx_max/8.9e-7))
  print("Interpolating\n" + str(gll) + "\nto\n" + str(cart))

# Apply the transformation
t_tmp = np.zeros((norder**2*ninterp,nelm), order='F')
t_tmp2 = np.zeros((norder*ninterp**2,nelm), order = 'F')
t_trans = np.zeros((ninterp**3,nelm), order = 'F')

# Transform to uniform grid
# z-first
tic()
t_p = np.reshape(np.transpose(np.reshape(t, (norder**2, norder, nelm), order='F'), (1,0,2)), (norder, norder**2*nelm), order='F')
t_tmp = np.reshape(np.transpose(np.reshape(trans.dot(t_p), (ninterp, norder**2, nelm), order='F'), (1,0,2)), (norder**2*ninterp, nelm), order='F')
toc('trans_z')

# then x
tic()
t_tmp2 = np.reshape(trans.dot(np.reshape(t_tmp, (norder, ninterp*norder*nelm), order='F')), (ninterp**2*norder,nelm), order='F')
toc('trans_x')

# then y
tic()
t_p =     np.reshape(np.transpose(np.reshape(t_tmp2,         (ninterp, norder, ninterp, nelm),  order='F'), (1,0,2,3)), (norder, ninterp**2*nelm), order='F')
t_trans = np.reshape(np.transpose(np.reshape(trans.dot(t_p), (ninterp, ninterp, ninterp, nelm), order='F'), (1,0,2,3)), (ninterp**3, nelm),        order='F')
toc('trans_y')

# Transform positions to uniform grid
tic()
pos_tmp = np.zeros((ninterp, ninterp, ninterp, 3), order='F')
pos_trans = np.zeros((ninterp**3, nelm, 3), order='F')
for i in range(nelm):
  for j in range(ninterp):
    pos_tmp[:,j,:,1] = pos[0,i,1] + cart[j]
    pos_tmp[j,:,:,0] = pos[0,i,0] + cart[j] 
    pos_tmp[:,:,j,2] = pos[0,i,2] + cart[j] 
  for j in range(3):
    pos_trans[:,i,j] = pos_tmp[:,:,:,j].flatten(order='F')
toc('trans_pos')

# extract for scatter plot
tic()
data = Grid(pos_trans, t_trans)
toc('to_grid')

#print(fractal_dimension(data))


# Renorm
Tt_low = -0.0005; Tt_high = 0.0005
data.f = (data.f - Tt_low)/(Tt_high - Tt_low)

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
  plot_slice(data, contour = cont)

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
  plt.savefig(fname+'_cdf.png')

plt.show()
toc('plot')

