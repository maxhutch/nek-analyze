#!/usr/bin/python3

import numpy as np
from sys import argv
import struct
import matplotlib.pyplot as plt
import scipy as sp
import scipy.linalg

print_timers = False

""" Timers from SO """
def tic():
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc(label):
    import time
    if 'startTime_for_tictoc' in globals():
      if print_timers:
        print("    > " + str(time.time() - startTime_for_tictoc) + "s in " + label)
    else:
        print("Toc: start time not set")

def find_root(x, y, y0 = .5, desired_resolution = None):
  from scipy import interpolate
  if desired_resolution == None:
    desired_resolution = abs(x[1] - x[0])/8.
  i_low = 0
  i_high = x.shape[0]
  failsafe = 0
  while (i_high - i_low) > 32 and failsafe < 10:
    failsafe += 1
    i_1 = int(i_low + (i_high-i_low)/3)
    i_2 = int(i_low + (i_high-i_low)*2/3)
    if y[i_1] > y0*1.1: 
      i_low = i_1
    if y[i_2] < y0*.9: 
      i_high = i_2

  f = interpolate.interp1d(x[i_low:i_high], y[i_low:i_high], kind='cubic') 
  x_low  = np.min(x[i_low:i_high])
  x_high = np.max(x[i_low:i_high])
  x_guess = (x_high + x_low)/2.
  while (x_high - x_low) > desired_resolution:
    fx = f(x_guess)
    if fx > y0:
      x_low = x_guess
    else:
      x_high = x_guess
    x_guess = (x_high + x_low)/2.
  return x_guess

class Grid:
  def __init__(self, shape):
    self.shape = shape
    return

  """ Unpack list of elements into single grid """
  def __init__(self, pos_elm, f_elm):
    from scipy.special import cbrt
    dx = pos_elm[1,0,0] - pos_elm[0,0,0]
    order = int(cbrt(pos_elm.shape[0]))
    origin = np.array([np.min(pos_elm[:,:,0]),
                       np.min(pos_elm[:,:,1]), 
                       np.min(pos_elm[:,:,2])])
    corner = np.array([np.max(pos_elm[:,:,0]), 
                       np.max(pos_elm[:,:,1]), 
                       np.max(pos_elm[:,:,2])])
    self.shape = np.array((corner - origin)/dx + .5, dtype=int)+1
    ''' position grid '''
    self.x = np.zeros((self.shape[0], self.shape[1], self.shape[2], 3), order='F') 
    for i in range(self.shape[0]):
      self.x[i,:,:,0] = dx*i + origin[0] 
    for i in range(self.shape[1]):
      self.x[:,i,:,1] = dx*i + origin[1] 
    for i in range(self.shape[2]):
      self.x[:,:,i,2] = dx*i + origin[2] 

    ''' field grid '''
    self.f = np.zeros((self.shape[0], self.shape[1], self.shape[2]), order='F') 
    for i in range(pos_elm.shape[1]):
      root = np.array((pos_elm[0,i,:] - origin)/dx + .5, dtype=int)
      self.f[root[0]:root[0]+order, 
             root[1]:root[1]+order,
             root[2]:root[2]+order] = np.reshape(f_elm[:,i], (order,order,order), order='F')

def plot_slice(grid, contour = None, fname = None):
  center = int(grid.shape[1]/2)
  plot_x = grid.x[:,center,:,0].ravel()
  plot_y = grid.x[:,center,:,2].ravel()
  plot_f = grid.f[:,center,:].ravel()


  fig = plt.figure(figsize=(12,12))
  ax1 = plt.subplot(1,1,1)
  plt.title('Grid Scatter plot')
  ax1.scatter(plot_x, plot_y, c=plot_f, s=2, linewidths = 0.)
  plt.axis([np.min(plot_x), np.max(plot_x), np.min(plot_y), np.max(plot_y)])
  if contour != None:
    ax1.plot(grid.x[:,center,0,0], contour, 'k-')
  plt.xlabel('X')
  plt.ylabel('Z')
  if fname != None:
    plt.savefig(fname)

def mixing_zone(grid, thresh = .1, plot = False, fname = None, time = -1.):
  f_xy = np.ones(data.shape[2])
  for i in range(data.shape[2]):
    f_xy[i] = np.average(data.f[:,:,i])

  boundaries = (find_root(data.x[1,1,:,2], f_xy, y0 = thresh), find_root(data.x[1,1,:,2], f_xy, y0 = 1-thresh))
  if fname != None:
    with open(fname, "a") as f:
      f.write("{:f} {:13.10f} {:13.10f}\n".format(time, boundaries[0], boundaries[1]))

  if plot: 
    plt.figure()
    ax1 = plt.subplot(1,1,1)
    ax1.plot(grid.x[1,1,:,2], f_xy, 'k-')
    ax1.vlines(boundaries, (0, 1), (thresh, 1-thresh))
    plt.savefig(argv[1]+'_mixing_zone.png')

  return boundaries 


""" Build Lagrange interpolation matrix """
def lagrange_matrix(A,B):
  M = np.zeros((B.size,A.size), order='F')
  for i in range(A.size):
    for j in range(B.size):
      M[j,i] =  1.
      for k in range(A.size):
        if k == i:
          continue
        M[j,i] = M[j,i] * (B[j] - A[k]) / (A[i] - A[k])
  return M

""" Load the data """
quiet = False

tic()
with open(argv[1],'rb') as f:
  #''' The header is 132 bytes long '''
  header = str(f.read(132))
  htoks = header.split()
  nelm = int(htoks[5])
  norder = int(htoks[2])
  time = float(htoks[7])
  #''' Assume isotropic elements '''
  ntot = nelm * norder * norder * norder
  if not quiet:
    print("Opened {:s} and found {:d} elements of order {:d}".format(argv[1], nelm, norder)) 
  #''' Check the test float '''
  test = struct.unpack('f', f.read(4))
  #print("  * test float is {}".format(test))
  #''' 4 byptes per element for an unused map '''
  element_map = f.read(nelm*4)
  #''' 4*3 bytes per basis function for position '''
  xyz  = np.fromfile(f, dtype=np.float32, count=ntot*3)
  #''' 4*3 bytes per basis function for velocity '''
  u    = np.fromfile(f, dtype=np.float32, count=norder*norder*norder*nelm*3)
  #''' 4 bytes per basis function for pressure '''
  p    = np.fromfile(f, dtype=np.float32, count=norder*norder*norder*nelm)
  #''' 4 bytes per basis function for temperature '''
  t_in = np.fromfile(f, dtype=np.float32, count=norder*norder*norder*nelm)

#''' Reshape vector data '''
pos = np.transpose(np.reshape(xyz, (norder*norder*norder,3,nelm), order='F'), (0,2,1))
vel = np.transpose(np.reshape(u, (norder*norder*norder,3,nelm), order='F'), (0,2,1))
#''' Reshape scaler data '''
t = np.reshape(t_in, (norder*norder*norder,nelm), order='F')
#''' Compute the total speed '''
speed = np.sqrt(np.square(vel[:,:,0]) + np.square(vel[:,:,1]) + np.square(vel[:,:,2]))
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
ninterp = int(norder*2)
gll  = pos[0:norder,0,0]
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

# Renorm
#Tt_low = np.min(t_trans); Tt_high = np.max(t_trans)
Tt_low = -0.0005; Tt_high = 0.0005
T_trans = (t_trans - Tt_low)/(Tt_high - Tt_low)

# extract for scatter plot
tic()
data = Grid(pos_trans, T_trans)
toc('to_grid')

cont = np.zeros((data.shape[0]))
tic()
center = data.shape[1]/2
for i in range(data.shape[0]):
  cont[i] = find_root(data.x[i,center,:,2], data.f[i,center,:])
toc('contour')

mixing_zone(data)

# Scatter plot of temperature (slice through pseudocolor in visit)
tic()
#plot_slice(data, contour = cont)
'''
# Fourier analysis in 1 dim
plt.figure()
bx1 = plt.subplot(1,2,1)
bx1.bar(  np.arange(int(data.shape[0]/2+1)),  abs(np.fft.rfft(data.f[:,center,int(data.shape[2]/2)])))
plt.title('temperature')
plt.xlabel('Mode')
plt.ylabel('Amplitude')
plt.xlim([0,10])
bx2 = plt.subplot(1,2,2)
bx2.bar(np.arange(int(data.shape[0]/2+1)),abs(np.fft.rfft(cont)))
plt.title('contour')
plt.xlabel('Mode')
plt.ylabel('Amplitude')
plt.xlim([0,10])

plt.savefig(argv[1]+'_spectrum.png')

plt.figure()
ax1 = plt.subplot(1,1,1)
ax1.hist(data.f.flatten(), bins=1000, normed=True, range=(-0.1,1.1), cumulative=True)
plt.xlim([-.1,1.1])
plt.savefig(argv[1]+'_cdf.png')

'''
plt.show()
toc('plot')

