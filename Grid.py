
Atwood = 1.e-3
g = 9.8

class Grid:
  def __init__(self, shape):
    self.shape = shape
    return

  """ Unpack list of elements into single grid """
  def __init__(self, order, origin, corner, shape, boxes = False):
    import numpy as np
    from threading import Lock
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
    self.pdf   = np.zeros(self.nbins)
    self.yind    = int(self.shape[1]/2. + .5)
    self.yslice  = np.zeros((self.shape[0], self.shape[2]), order='F')
    self.zind    = int(self.shape[2]/2. + .5)
    self.zslice  = np.zeros((self.shape[0], self.shape[1]), order='F')
    self.zsliceu = np.zeros((self.shape[0], self.shape[1],3), order = 'F')
    self.dotzsliceu = np.zeros((self.shape[0], self.shape[1]), order = 'F')
    self.boxes = None
    self.interface = None
    if boxes:
      self.boxes = np.zeros(int(np.log2(np.min(self.shape))))
      self.interface = np.zeros(np.prod(self.shape / self.order), dtype=np.bool_)
    self.lock = Lock()

  def add(self, pos_elm, f_elm, ux_elm, uy_elm, uz_elm):
    import numpy as np
    import numpy.linalg as lin
    import scipy.ndimage.measurements as measurements
    from my_utils import compute_index
    from tictoc import tic, toc
    '''
    if self.f == None:
      self.f  = np.zeros((self.shape[0], self.shape[1], self.shape[2]), order='F', dtype=np.float64)
      self.ux = np.zeros((self.shape[0], self.shape[1], self.shape[2]), order='F', dtype=np.float64)
      self.uy = np.zeros((self.shape[0], self.shape[1], self.shape[2]), order='F', dtype=np.float64)
      self.uz = np.zeros((self.shape[0], self.shape[1], self.shape[2]), order='F', dtype=np.float64)
    '''

    tic()
    tmp = np.sum(np.minimum(f_elm*2., (1.-f_elm)*2.))
    with self.lock:
      self.f_m += tmp
    tmp = np.sum(np.square(ux_elm)) + np.sum(np.square(uy_elm)) + np.sum(np.square(uz_elm))
    with self.lock:
      self.v2  += tmp
    pdf_partial, foo = np.histogram(f_elm.ravel(), bins=self.nbins, range=(-0.1, 1.1))
    with self.lock:
      self.pdf += pdf_partial
    toc('aggregate')

    if self.boxes != None:
      tic()
      f_tmp  = np.reshape(f_elm, (self.order,self.order,self.order*f_elm.shape[1]), order='F')
      fs = np.sign(f_tmp-0.5)
      X, Y, Z = np.split(np.mgrid[0:self.order, 0:self.order, 0:self.order*f_elm.shape[1]].astype(int), 3, axis=0)
      for j in range(1,int(np.log2(self.order))+1):
        fac = int(2**j)
        regions = np.array((X/fac), dtype=int) \
                + np.array((Y/fac), dtype=int) * (self.order/fac) \
                + np.array((Z/fac), dtype=int) * (self.order/fac)**2 
        N = (self.order/fac)**3 * f_elm.shape[1]
        interface = np.multiply(
                     measurements.minimum(fs, labels=regions, index=np.arange(N)), 
                     measurements.maximum(fs, labels=regions, index=np.arange(N))
                    ) 
        self.boxes[j-1] += (N - np.sum(interface))/2
      toc('box1')

    for i in range(pos_elm.shape[0]):
      root = np.array((pos_elm[i,:] - self.origin)/self.dx + .5, dtype=int)
      if self.interface != None:
        foo = False
        if np.max(f_elm[:,i]-.5) * np.min(f_elm[:,i]-.5) < 0:
          foo = True
        key = np.array([2,2,2], dtype=int)
        self.interface[compute_index((root / self.order) + key, self.shape/self.order)] = foo

      yoff = self.yind - root[1]
      zoff = self.zind - root[2]
      f_tmp  = np.reshape(f_elm[:,i], (self.order,self.order,self.order), order='F')

      self.f_xy[root[2]:root[2]+self.order] += np.sum(f_tmp, (0,1))
      if yoff >= 0 and yoff < self.order:
        self.yslice[root[0]:root[0]+self.order, 
                    root[2]:root[2]+self.order] = f_tmp[:,yoff,:]
      if zoff >= 0 and zoff < self.order:
        ux_tmp = np.reshape(ux_elm[:,i], (self.order,self.order,self.order), order='F')
        uy_tmp = np.reshape(uy_elm[:,i], (self.order,self.order,self.order), order='F')
        uz_tmp = np.reshape(uz_elm[:,i], (self.order,self.order,self.order), order='F')
        self.zslice[root[0]:root[0]+self.order, 
                    root[1]:root[1]+self.order] = f_tmp[:,:,zoff]
        self.zsliceu[root[0]:root[0]+self.order, 
                     root[1]:root[1]+self.order, 0] = ux_tmp[:,:,zoff]
        self.zsliceu[root[0]:root[0]+self.order, 
                     root[1]:root[1]+self.order, 1] = uy_tmp[:,:,zoff]
        self.zsliceu[root[0]:root[0]+self.order, 
                     root[1]:root[1]+self.order, 2] = uz_tmp[:,:,zoff]
        self.dotzsliceu[root[0]:root[0]+self.order, 
                        root[1]:root[1]+self.order] = (uz_tmp[:,:,zoff+1] - uz_tmp[:,:,zoff-1])/(2.*self.dx)
        '''
        self.f[root[0]:root[0]+self.order,
               root[1]:root[1]+self.order,
               root[2]:root[2]+self.order] = f_tmp
        '''
        '''
        self.ux[root[0]:root[0]+self.order,
                root[1]:root[1]+self.order,
                root[2]:root[2]+self.order] = u_tmp
        '''
        '''
        self.uy[root[0]:root[0]+self.order,
                root[1]:root[1]+self.order,
                root[2]:root[2]+self.order] = u_tmp 
        '''
        '''
        self.uz[root[0]:root[0]+self.order,
                root[1]:root[1]+self.order,
                root[2]:root[2]+self.order] = u_tmp
        '''
 
  def add_pos(self):
    import numpy as np
    ''' position grid '''
    self.x = np.zeros((self.shape[0], self.shape[1], self.shape[2], 3), order='F', dtype=np.float64)
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
  ax1.imshow(grid.yslice.transpose(), origin = 'lower', interpolation='bicubic')
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

  modes_x = np.fft.fftfreq( grid.shape[0], grid.dx)
  modes_y = np.fft.rfftfreq(grid.shape[1], grid.dx)
  modes = np.zeros((modes_x.size, modes_y.size))
  for i in range(modes_x.size):
    for j in range(modes_y.size):
      modes[i,j] = np.sqrt(abs(modes_x[i])*abs(modes_x[i]) + abs(modes_y[j]) * abs(modes_y[j]))
  plt.xlim([modes_y[1]/1.5, np.max(modes)*1.5])

  for zpos in slices:
    z = int(zpos * grid.shape[2])

    # compute Taylor microscale
    spectrum = np.square(np.abs(np.fft.rfft2(grid.zslice[:,:])) / (grid.shape[0]*grid.shape[1]))
    ax1.plot(modes.ravel(), .25 * Atwood * g * spectrum.ravel(), 'bo', label='P')

    spectrum_uz = np.square(np.abs(np.fft.rfft2(grid.zsliceu[:,:,2]))/ (grid.shape[0]*grid.shape[1]))
    spectrum_xy = np.square(np.abs(np.fft.rfft2(grid.zsliceu[:,:,0])) / (grid.shape[0]*grid.shape[1]))
    spectrum_xy += np.square(np.abs(np.fft.rfft2(grid.zsliceu[:,:,1])) / (grid.shape[0]*grid.shape[1]))
    ax1.plot(modes.ravel(), .5*(spectrum_uz + spectrum_xy).ravel(), 'bx', label='K')
    ax1.plot(modes.ravel(), .5*(spectrum_uz.ravel()     ), 'rx', label='K_z')
    ax1.plot(modes.ravel(), .5*(spectrum_xy.ravel()     ), 'gx', label='K_xy')

    tmp = np.average(np.square((grid.dotzsliceu[:,:]))) 
    taylor_z = np.sqrt(np.average(np.square(grid.zsliceu[:,:,2])) / tmp)
    kolmog_z = ((8.9e-7)**2 / (15. * tmp))**(1./4.)
    ax1.vlines(1./taylor_z, 1.e-20, 1., label='lambda_z', color='r', linestyles='dashdot')
    ax1.vlines(1./kolmog_z, 1.e-20, 1., label='eta_z', color='r', linestyles='dotted')
    tmp = np.average(np.square((grid.zsliceu[:,2:,1] - grid.zsliceu[:,:-2,1])/(2.*grid.dx))) 
    taylor_y = np.sqrt(np.average(np.square(grid.zsliceu[:,1:-1,1])) / tmp)
    kolmog_y = ((8.9e-7)**2 / (15. * tmp))**(1./4.)

    tmp = np.average(np.square((grid.zsliceu[2:,:,0] - grid.zsliceu[:-2,:,0])/(2.*grid.dx))) 
    taylor_x = np.sqrt(np.average(np.square(grid.zsliceu[1:-1,:,0])) / tmp)
    kolmog_x = ((8.9e-7)**2 / (15. * tmp))**(1./4.)

    ax1.vlines(2./(taylor_x+taylor_y), 1.e-20, 1., label='lambda_xy', color='g', linestyles='dashdot')
    ax1.vlines(2./(kolmog_z+kolmog_y), 1.e-20, 1., label='eta_xy', color='g', linestyles='dotted')

  plt.legend(loc=3)
  
  xs = np.sort(modes.ravel())
  ys = xs**(-5./3.) 
  for i in range(9):
    ax1.plot(xs, 10**(1.-2.*i) * ys, 'k--')

  if fname != None:
    plt.savefig(fname)

  '''
  if contour:
    plt.figure(figsize=(12,12))
    ax1 = plt.subplot(1,1,1)
    plt.title('Interface Spectrum')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Mode Number')
    spectrum = np.fft.rfft2(grid.cont[:,:]) / (grid.shape[0]*grid.shape[1])
    ax1.plot(modes.ravel(), spectrum.ravel(), '+', label='I')
  '''

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

  X = float(grid.f_m/(h*grid.shape[0]*grid.shape[1]))

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

