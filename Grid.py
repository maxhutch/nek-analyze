
Atwood = 1.e-3
g = 9.8

class Grid:
  def __init__(self, shape):
    self.shape = shape
    return

  """ Unpack list of elements into single grid """
  def __init__(self, order, origin, corner, shape):
    import numpy as np
    self.order = order 
    self.origin = origin
    self.corner = corner 
    self.shape = shape 
    self.dx = (self.corner[0] - self.origin[0])/(self.shape[0])
    self.f, self.ux, self.uy, self.uz = None, None, None, None
    self.f_xy = np.zeros(self.shape[2])
    self.f_m  = 0.
    self.v2   = 0.
    self.nbins = 1000
    self.pdf  = np.zeros(self.nbins)

  def add(self, pos_elm, f_elm = None, ux_elm = None, uy_elm = None, uz_elm = None):
    import numpy as np
    import gc
    from memory import resident
    ''' field grid '''
    if f_elm != None:
      if self.f == None:
        self.f = np.zeros((self.shape[0], self.shape[1], self.shape[2]), order='F', dtype=np.float32)
      self.f_m += np.sum(np.where(f_elm[:,:] < .5, f_elm[:,:]*2, 2*(1.-f_elm[:,:])))
      pdf_partial, foo = np.histogram(f_elm.flatten(), bins=self.nbins, range=(-0.1, 1.1))
      self.pdf += pdf_partial
      for i in range(pos_elm.shape[1]):
        root = np.array((pos_elm[0,i,:] - self.origin)/self.dx + .5, dtype=int)
        f_tmp = np.reshape(f_elm[:,i], (self.order,self.order,self.order), order='F')
        self.f_xy[root[2]:root[2]+self.order] += np.sum(f_tmp, (0,1))
        self.f[root[0]:root[0]+self.order,
               root[1]:root[1]+self.order,
               root[2]:root[2]+self.order] = f_tmp

    ''' field grid '''
    if ux_elm != None:
      if self.ux == None:
        self.ux = np.zeros((self.shape[0], self.shape[1], self.shape[2]), order='F', dtype=np.float32)
      self.v2 += np.sum(np.square(ux_elm))
      for i in range(pos_elm.shape[1]):
        root = np.array((pos_elm[0,i,:] - self.origin)/self.dx + .5, dtype=int)
        self.ux[root[0]:root[0]+self.order,
                root[1]:root[1]+self.order,
                root[2]:root[2]+self.order] = np.reshape(ux_elm[:,i], (self.order,self.order,self.order), order='F')

    ''' field grid '''
    if uy_elm != None:
      if self.uy == None:
        self.uy = np.zeros((self.shape[0], self.shape[1], self.shape[2]), order='F', dtype=np.float32)
      self.v2 += np.sum(np.square(uy_elm))
      for i in range(pos_elm.shape[1]):
        root = np.array((pos_elm[0,i,:] - self.origin)/self.dx + .5, dtype=int)
        self.uy[root[0]:root[0]+self.order,
                root[1]:root[1]+self.order,
                root[2]:root[2]+self.order] = np.reshape(uy_elm[:,i], (self.order,self.order,self.order), order='F')

    ''' field grid '''
    if uz_elm != None:
      if self.uz == None:
        self.uz = np.zeros((self.shape[0], self.shape[1], self.shape[2]), order='F', dtype=np.float32)
      self.v2 += np.sum(np.square(uz_elm))
      for i in range(pos_elm.shape[1]):
        root = np.array((pos_elm[0,i,:] - self.origin)/self.dx + .5, dtype=int)
        self.uz[root[0]:root[0]+self.order,
                root[1]:root[1]+self.order,
                root[2]:root[2]+self.order] = np.reshape(uz_elm[:,i], (self.order,self.order,self.order), order='F')

  def add_pos(self):
    import numpy as np
    ''' position grid '''
    self.x = np.zeros((self.shape[0], self.shape[1], self.shape[2], 3), order='F', dtype=np.float32)
    for i in range(self.shape[0]):
      self.x[i,:,:,0] = self.dx*i + self.origin[0]
    for i in range(self.shape[1]):
      self.x[:,i,:,1] = self.dx*i + self.origin[1]
    for i in range(self.shape[2]):
      self.x[:,:,i,2] = self.dx*i + self.origin[2]

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

def plot_dist(grid, fname = None):
  import matplotlib.pyplot as plt
  import numpy as np
  edges = np.linspace(-0.1, 1.1, grid.nbins+1)
  grid.pdf = grid.pdf / np.sum(grid.pdf)
  cdf = np.cumsum(grid.pdf)
  plt.figure()
  ax1 = plt.subplot(1,1,1)
  ax1.bar(edges[:-1], cdf, width=(edges[1]-edges[0]))
  plt.xlim([-.1,1.1])
  plt.ylim([0,1])
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

  L = np.max(grid.corner[2]) - np.min(grid.origin[2])

  f_xy = np.ones(grid.shape[2])
  for i in range(grid.shape[2]):
    f_xy[i] = grid.f_xy[i] / (grid.shape[0]*grid.shape[1]) 

  # Cabot's h
  h = 0.
  for i in range(f_xy.shape[0]):
    if f_xy[i] < .5:
      h += 2*f_xy[i]
    else:
      h += 2*(1.-f_xy[i])
  h_cabot = (h * L / f_xy.shape[0])

  # visual h
  zs = np.linspace(grid.origin[2], grid.corner[2], grid.shape[2], endpoint = False)
  h_visual = ( find_root(zs, f_xy, y0 = thresh) 
             - find_root(zs, f_xy, y0 = 1-thresh)) / 2.

  X = grid.f_m/(h*grid.shape[0]*grid.shape[1])

  return h_cabot, h_visual, X

def energy_budget(grid):
  import numpy as np
  from my_utils import find_root

  # Potential
  zs = np.linspace(grid.origin[2], grid.corner[2], grid.shape[2], endpoint = False)
  dV = grid.dx * grid.dx * grid.dx 
  U = 0.
  for i in range(grid.shape[2]):
    U = U - grid.f_xy[i] * zs[i] * dV
  U0 = np.prod(grid.shape)/2. * dV * zs[int(grid.shape[2]*3./4.)]

  # Kinetic
  K = grid.v2 * dV/2.
  return Atwood*g*(U0 - U), K

