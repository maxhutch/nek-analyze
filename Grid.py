
Atwood = 1.e-3
g = 9.8

class Grid:
  def __init__(self, shape):
    self.shape = shape
    return

  """ Unpack list of elements into single grid """
  def __init__(self, pos_elm, f_elm, ux_elm = None, uy_elm = None, uz_elm = None, speed_elm = None):
    import numpy as np
    from scipy.special import cbrt
    import gc
    dx = pos_elm[1,0,0] - pos_elm[0,0,0]
    order = int(cbrt(pos_elm.shape[0]))
    origin = np.array([np.min(pos_elm[:,:,0]),
                       np.min(pos_elm[:,:,1]),
                       np.min(pos_elm[:,:,2])])
    corner = np.array([np.max(pos_elm[:,:,0]),
                       np.max(pos_elm[:,:,1]),
                       np.max(pos_elm[:,:,2])])
    self.shape = np.array((corner - origin)/dx + .5, dtype=int)+1

    ''' field grid '''
    self.f = np.zeros((self.shape[0], self.shape[1], self.shape[2]), order='F')
    for i in range(pos_elm.shape[1]):
      root = np.array((pos_elm[0,i,:] - origin)/dx + .5, dtype=int)
      self.f[root[0]:root[0]+order,
             root[1]:root[1]+order,
             root[2]:root[2]+order] = np.reshape(f_elm[:,i], (order,order,order), order='F')
    f_elm = None; gc.collect()

    ''' field grid '''
    if ux_elm != None:
      self.ux = np.zeros((self.shape[0], self.shape[1], self.shape[2]), order='F')
      for i in range(pos_elm.shape[1]):
        root = np.array((pos_elm[0,i,:] - origin)/dx + .5, dtype=int)
        self.ux[root[0]:root[0]+order,
               root[1]:root[1]+order,
               root[2]:root[2]+order] = np.reshape(ux_elm[:,i], (order,order,order), order='F')
    else:
      self.ux = None
    ux_elm = None; gc.collect()

    ''' field grid '''
    if uy_elm != None:
      self.uy = np.zeros((self.shape[0], self.shape[1], self.shape[2]), order='F')
      for i in range(pos_elm.shape[1]):
        root = np.array((pos_elm[0,i,:] - origin)/dx + .5, dtype=int)
        self.uy[root[0]:root[0]+order,
               root[1]:root[1]+order,
               root[2]:root[2]+order] = np.reshape(uy_elm[:,i], (order,order,order), order='F')
    else:
      self.uy = None
    uy_elm = None; gc.collect()


    ''' field grid '''
    if uz_elm != None:
      self.uz = np.zeros((self.shape[0], self.shape[1], self.shape[2]), order='F')
      for i in range(pos_elm.shape[1]):
        root = np.array((pos_elm[0,i,:] - origin)/dx + .5, dtype=int)
        self.uz[root[0]:root[0]+order,
               root[1]:root[1]+order,
               root[2]:root[2]+order] = np.reshape(uz_elm[:,i], (order,order,order), order='F')
    else:
      self.uz = None
    uz_elm = None; gc.collect()
    pos_elm = None; gc.collect()

    ''' field grid '''
    if self.ux != None and self.uy != None and self.uz != None:
      self.speed = np.sqrt(np.square(self.ux) + np.square(self.uy) + np.square(self.uz))
    else:
      self.speed = None

    ''' position grid '''
    self.x = np.zeros((self.shape[0], self.shape[1], self.shape[2], 3), order='F')
    for i in range(self.shape[0]):
      self.x[i,:,:,0] = dx*i + origin[0]
    for i in range(self.shape[1]):
      self.x[:,i,:,1] = dx*i + origin[1]
    for i in range(self.shape[2]):
      self.x[:,:,i,2] = dx*i + origin[2]

def covering_number(grid, N):
  import numpy as np
  N = int(np.min(grid.shape / N))
  nshift = int(N/2)+1
  ans = grid.f.size
  for ii in range(0,N,nshift):
   for jj in range(0,N,nshift):
    for kk in range(0,N,nshift):
     counter = 0
     for i in range(ii,grid.shape[0],N): 
      for j in range(jj,grid.shape[1],N): 
       for k in range(kk,grid.shape[2],N): 
        box = grid.f[i:i+N+1,j:j+N+1,k:k+N+1]
        if np.max(box) * np.min(box) <= 0.:
          counter+=1
     ans = min(ans, counter)
  return ans

def fractal_dimension(grid, nsample = 25, base = 1.2):
  import numpy as np
  from scipy.stats import linregress
  cover_number = []
  nbox = []
  for i in range(1,nsample):
    n = int(base**i)
    if len(nbox) > 0 and n == nbox[-1]:
      continue
    nbox.append(n)
    cover_number.append(covering_number(grid, n))
  nbox = np.array(nbox)
  cover_number = np.array(cover_number)
  ans = linregress(np.log(nbox), np.log(cover_number))
  return ans[0]

def plot_slice(grid, fname = None):
  import numpy as np 
  import matplotlib.pyplot as plt
  center = int(grid.shape[1]/2)

  image_x = 12
  image_y = int(image_x * grid.shape[2] / grid.shape[0] + .5)
  fig = plt.figure(figsize=(image_x,image_y))
  ax1 = plt.subplot(1,1,1)
  plt.title('Y-normal slice')
  ax1.imshow(grid.f[:,center,:].transpose(), origin = 'lower', interpolation='bicubic')
  plt.xlabel('X')
  plt.ylabel('Z')
  if fname != None:
    plt.savefig(fname)

def plot_spectrum(grid, fname = None, slices = None, contour = False):
  import numpy as np 
  import matplotlib.pyplot as plt
  if slices == None:
    slices = [.5]

  plt.figure(figsize=(12,12))
  ax1 = plt.subplot(1,1,1)
  plt.title('Energy Spectrum')
  plt.xscale('log')
  plt.yscale('log')
  plt.xlabel('Mode Number')
  plt.ylabel('Amplitude')
  plt.ylim([10**(-20),.25*Atwood*g])

  modes_x = np.fft.fftfreq(grid.shape[0], grid.x[1,0,0,0] - grid.x[0,0,0,0])
  modes_y = np.fft.rfftfreq(grid.shape[1], grid.x[0,1,0,1] - grid.x[0,0,0,1])
  modes = np.zeros((modes_x.size, modes_y.size))
  for i in range(modes_x.size):
    for j in range(modes_y.size):
      modes[i,j] = np.sqrt(abs(modes_x[i])*abs(modes_x[i]) + abs(modes_y[j]) * abs(modes_y[j]))
  plt.xlim([modes_y[1]/1.5, np.max(modes)*1.5])
  for zpos in slices:
    z = int(zpos * grid.shape[2])

    # compute Taylor microscale
    spectrum = np.fft.rfft2(grid.f[:,:,z]) / (grid.shape[0]*grid.shape[1])
    ax1.plot(modes.ravel(), .25 * Atwood * g * np.square(np.abs(spectrum.ravel())), 'o', label='P')
    if grid.uz != None:
      tmp = np.average(np.square((grid.uz[:,:,z+1] - grid.uz[:,:,z-1])/(grid.x[0,0,z+1,2] - grid.x[0,0,z-1,2]))) 
      taylor_z = np.sqrt(np.average(np.square(grid.uz[:,:,z])) / tmp)
      kolmog_z = ((8.9e-7)**2 / (15. * tmp))**(1./4.)
      spectrum = np.square(np.abs(np.fft.rfft2(grid.uz[:,:,z]))/ (grid.shape[0]*grid.shape[1]))
      ax1.plot(modes.ravel(), .5*spectrum.ravel(), 'x', label='K_z')
      ax1.vlines(1./taylor_z, 1.e-20, 1., label='lambda_z', color='g', linestyles='dashdot')
      ax1.vlines(1./kolmog_z, 1.e-20, 1., label='eta_z', color='g', linestyles='dotted')
    if grid.ux != None and grid.uy != None and grid.uz != None:
      tmp = np.average(np.square((grid.uy[:,2:,z] - grid.uy[:,:-2,z])/(grid.x[0,2,z,1] - grid.x[0,0,z,1]))) 
      taylor_y = np.sqrt(np.average(np.square(grid.uy[:,1:-1,z])) / tmp)
      kolmog_y = ((8.9e-7)**2 / (15. * tmp))**(1./4.)

      tmp = np.average(np.square((grid.ux[2:,:,z] - grid.ux[:-2,:,z])/(grid.x[2,0,z,0] - grid.x[0,0,z,0]))) 
      taylor_x = np.sqrt(np.average(np.square(grid.uy[1:-1,:,z])) / tmp)
      kolmog_x = ((8.9e-7)**2 / (15. * tmp))**(1./4.)

      spectrum += np.square(np.abs(np.fft.rfft2(grid.ux[:,:,z])) / (grid.shape[0]*grid.shape[1]))
      spectrum += np.square(np.abs(np.fft.rfft2(grid.uy[:,:,z])) / (grid.shape[0]*grid.shape[1]))
      ax1.plot(modes.ravel(), .5*spectrum.ravel(), '+', label='K')
      ax1.vlines(2./(taylor_x+taylor_y), 1.e-20, 1., label='lambda_xy', color='y', linestyles='dashdot')
      ax1.vlines(2./(kolmog_z+kolmog_y), 1.e-20, 1., label='eta_xy', color='y', linestyles='dotted')

  plt.legend(loc=3)
  
  xs = np.sort(modes.ravel())
  ys = xs**(-5./3.) 
  for i in range(9):
    ax1.plot(xs, 10**(1.-2.*i) * ys, 'k--')

  if fname != None:
    plt.savefig(fname)

  if contour:
    plt.figure(figsize=(12,12))
    ax1 = plt.subplot(1,1,1)
    plt.title('Interface Spectrum')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Mode Number')
    spectrum = np.fft.rfft2(grid.cont[:,:]) / (grid.shape[0]*grid.shape[1])
    ax1.plot(modes.ravel(), spectrum.ravel(), '+', label='I')

def mixing_zone(grid, thresh = .01):
  import numpy as np
  from my_utils import find_root

  L = np.max(grid.x[1,1,:,2]) - np.min(grid.x[1,1,:,2])

  f_xy = np.ones(grid.shape[2])
  for i in range(grid.shape[2]):
    f_xy[i] = np.average(grid.f[:,:,i])

  # Cabot's h
  h = 0.
  for i in range(f_xy.shape[0]):
    if f_xy[i] < .5:
      h += 2*f_xy[i]
    else:
      h += 2*(1.-f_xy[i])
  h_cabot = (h * L / f_xy.shape[0])

  # visual h
  h_visual = ( find_root(grid.x[1,1,:,2], f_xy, y0 = thresh) 
             - find_root(grid.x[1,1,:,2], f_xy, y0 = 1-thresh)) / 2.

  f_m = np.where(grid.f[:,:,:] < .5, grid.f[:,:,:]*2, 2*(1.-grid.f[:,:,:]))
  X = np.average(f_m)*grid.shape[2]/h

  return h_cabot, h_visual, X

def energy_budget(grid):
  import numpy as np
  from my_utils import find_root

  # Potential
  dV = np.prod(grid.x[1,1,1,:]-grid.x[0,0,0,:])
  U = 0.
  for i in range(grid.shape[2]):
    U = U - np.sum(grid.f[:,:,i]) * grid.x[0,0,i,2] * dV
  U0 = np.prod(grid.shape)/2. * dV *grid.x[0,0,int(grid.shape[2]*3./4.),2]

  # Kinetic
  K = np.sum(np.square(grid.speed))*dV/2.
  return Atwood*g*(U0 - U), K

