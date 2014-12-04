
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

    # Basic mesh info: do not change 
    self.order = order 
    self.origin = np.array(origin)
    self.corner = np.array(corner)
    self.shape = shape 
    self.dx = (self.corner[:] - self.origin[:])/(self.shape[:])
    self.f, self.ux, self.uy, self.uz = None, None, None, None

    # Aggregate quantities with inits
    self.nbins = 1000
    self.f_total  = 0.
    self.f_m      = 0.
    self.v2       = 0. 
    self.pdf      = np.zeros(self.nbins)

    # Slice information (add anisotropy tensor (<u_i u_j>/<u_k u_k>)
    self.f_xy       = np.zeros(self.shape[2])
    self.ff_xy      = np.zeros(self.shape[2])
    self.vv_xy      = np.zeros((self.shape[2],6), order='F')
    self.yind       = int(3*self.shape[1]/4. + .5)
    self.yslice     = np.zeros((self.shape[0], self.shape[2]), order='F')
    self.ypslice    = np.zeros((self.shape[0], self.shape[2]), order='F')
    self.yuzslice   = np.zeros((self.shape[0], self.shape[2]), order='F')
    self.yuxslice   = np.zeros((self.shape[0], self.shape[2]), order='F')
    self.zind       = int(self.shape[2]/2. + .5)
    self.zslice     = np.zeros((self.shape[0], self.shape[1]), order='F')
    self.zsliceu    = np.zeros((self.shape[0], self.shape[1],3), order = 'F')
    self.dotzsliceu = np.zeros((self.shape[0], self.shape[1]), order = 'F')

    # Fractal dimension stuff
    self.box_pos  = None
    self.box_dist = None
    self.nbox = 1000
    if boxes:
      self.box_pos  = (np.random.randn(3, self.nbox) + 3.)/6.
      self.box_pos[0,:] = self.box_pos[0,:] * (self.corner[0] - self.origin[0]) + self.origin[0]
      self.box_pos[1,:] = self.box_pos[1,:] * (self.corner[1] - self.origin[1]) + self.origin[1]
      self.box_pos[2,:] = self.box_pos[2,:] * (self.corner[2] - self.origin[2]) + self.origin[2]
      self.box_dist = np.ones((2, self.nbox), order = 'F') * np.linalg.norm(self.corner - self.origin) 

  def merge(self, part):
    self.f_total     += part.f_total
    self.f_m         += part.f_m
    self.v2          += part.v2
    self.pdf         += part.pdf
    self.f_xy        += part.f_xy
    self.ff_xy       += part.ff_xy
    self.vv_xy       += part.vv_xy
    self.yslice      += part.yslice
    self.ypslice     += part.ypslice
    self.yuzslice    += part.yuzslice
    self.yuxslice    += part.yuxslice
    self.zslice      += part.zslice
    self.zsliceu     += part.zsliceu
    self.dotzsliceu  += part.dotzsliceu

  def add(self, pos_elm, p_elm, f_elm, ux_elm, uy_elm, uz_elm):
    import numpy as np
    import numpy.linalg as lin
    import scipy.ndimage.measurements as measurements
    from my_utils import compute_index
    import time as timer
    from tictoc import tic, toc

    # First, compute aggregate quantities like total kinetic energy
    tic()
    self.f_total  = np.add.reduce(f_elm, axis=None)
    self.f_m      = np.add.reduce(np.minimum(f_elm*2., (1.-f_elm)*2.), axis=None)
    self.v2       = (np.add.reduce(np.square(ux_elm), axis=None) 
                   + np.add.reduce(np.square(uy_elm), axis=None)
                   + np.add.reduce(np.square(uz_elm), axis=None)
                    )
    self.pdf, foo = np.histogram(f_elm.ravel(), bins=self.nbins, range=(-0.1, 1.1))
    toc('aggregate')

    # element-wise operations and slices
    self.f_xy       = np.zeros(self.shape[2])
    self.ff_xy      = np.zeros(self.shape[2])
    self.yslice     = np.zeros((self.shape[0], self.shape[2]), order='F')
    self.ypslice    = np.zeros((self.shape[0], self.shape[2]), order='F')
    self.yuzslice   = np.zeros((self.shape[0], self.shape[2]), order='F') 
    self.yuxslice   = np.zeros((self.shape[0], self.shape[2]), order='F') 
    self.zslice     = np.zeros((self.shape[0], self.shape[1]), order='F')
    self.zsliceu    = np.zeros((self.shape[0], self.shape[1],3), order = 'F')
    self.dotzsliceu = np.zeros((self.shape[0], self.shape[1]), order = 'F')

    for i in range(pos_elm.shape[1]):
      root = np.array((pos_elm[:,i] - self.origin)/self.dx + .5, dtype=int)
      yoff = self.yind - root[1]
      zoff = self.zind - root[2]
      f_tmp  = np.reshape(f_elm[:,i], (self.order,self.order,self.order), order='F')
      ux_tmp = np.reshape(ux_elm[:,i], (self.order,self.order,self.order), order='F')
      uy_tmp = np.reshape(uy_elm[:,i], (self.order,self.order,self.order), order='F')
      uz_tmp = np.reshape(uz_elm[:,i], (self.order,self.order,self.order), order='F')

      self.f_xy[ root[2]:root[2]+self.order]   += np.add.reduce(f_tmp, (0,1))
      self.ff_xy[ root[2]:root[2]+self.order]  += np.add.reduce(f_tmp*(1.-f_tmp), (0,1))
      self.vv_xy[root[2]:root[2]+self.order,0] += np.add.reduce(ux_tmp*ux_tmp, (0,1))
      self.vv_xy[root[2]:root[2]+self.order,1] += np.add.reduce(ux_tmp*uy_tmp, (0,1))
      self.vv_xy[root[2]:root[2]+self.order,2] += np.add.reduce(ux_tmp*uz_tmp, (0,1))
      self.vv_xy[root[2]:root[2]+self.order,3] += np.add.reduce(uy_tmp*uy_tmp, (0,1))
      self.vv_xy[root[2]:root[2]+self.order,4] += np.add.reduce(uy_tmp*uz_tmp, (0,1))
      self.vv_xy[root[2]:root[2]+self.order,5] += np.add.reduce(uz_tmp*uz_tmp, (0,1))

      if yoff >= 0 and yoff < self.order:
        p_tmp = np.reshape(p_elm[:,i], (self.order,self.order,self.order), order='F')

        # y-slice of scalar field
        self.yslice[  root[0]:root[0]+self.order, 
                      root[2]:root[2]+self.order] = f_tmp[:,yoff,:]
        self.yuzslice[root[0]:root[0]+self.order, 
                      root[2]:root[2]+self.order] = uz_tmp[:,yoff,:]
        self.yuxslice[root[0]:root[0]+self.order, 
                      root[2]:root[2]+self.order] = ux_tmp[:,yoff,:]

        #mgh = 1.e-3 * 9.8*(np.tile(np.arange(0,self.order)*self.dx, (self.order,1)))
        #mgh += 1.e-3 * 9.8 * pos_elm[2,i]

        self.ypslice[root[0]:root[0]+self.order, 
                     root[2]:root[2]+self.order] = p_tmp[:,yoff,:]# - mgh * f_tmp[:,yoff,:]

      if zoff >= 0 and zoff < self.order:
        self.zslice[root[0]:root[0]+self.order, 
                    root[1]:root[1]+self.order] = f_tmp[:,:,zoff]
        self.zsliceu[root[0]:root[0]+self.order, 
                     root[1]:root[1]+self.order, 0] = ux_tmp[:,:,zoff]
        self.zsliceu[root[0]:root[0]+self.order, 
                     root[1]:root[1]+self.order, 1] = uy_tmp[:,:,zoff]
        self.zsliceu[root[0]:root[0]+self.order, 
                     root[1]:root[1]+self.order, 2] = uz_tmp[:,:,zoff]
        self.dotzsliceu[root[0]:root[0]+self.order, 
                        root[1]:root[1]+self.order] = (uz_tmp[:,:,zoff+1] - uz_tmp[:,:,zoff-1])/(2.*self.dx[2])

    # box counting
    if self.box_pos != None:
      tic()
      fs= np.sign(np.reshape(f_elm, (self.order,self.order,self.order,f_elm.shape[1]), order='F') - 0.5)
      X, Y, Z = np.split(np.mgrid[0:self.order, 0:self.order, 0:self.order]*self.dx[0], 3, axis=0)
      pad = self.order*self.dx*np.sqrt(3.) 
      toc('prework')
      sort_time = 0.
      search_time = 0.
      for j in range(self.box_pos.shape[1]):
        sort_time -= timer.time()
        elm_dists = np.linalg.norm(self.box_pos[:,j,np.newaxis] - pos_elm[:,:], axis=0)
        #sorted_indices = np.argsort(elm_dists)
        nsort = 10
        sorted_indices = np.argpartition(elm_dists, np.arange(nsort))[:nsort]
        sort_time += timer.time()

        search_time -= timer.time()
        done = 0
        for i in sorted_indices:
          if max(self.box_dist[0,j], self.box_dist[1,j]) < elm_dists[i] - pad:
            done = 1
            break 
          dist = np.sqrt(
                         np.square(pos_elm[0,i] + X - self.box_pos[0,j]) 
                       + np.square(pos_elm[1,i] + Y - self.box_pos[1,j]) 
                       + np.square(pos_elm[2,i] + Z - self.box_pos[2,j]) 
                        )
          tmp1 = np.amin(np.where(fs[:,:,:,i]>0, dist, np.inf))
          tmp2 = np.amin(np.where(fs[:,:,:,i]<0, dist, np.inf))
          #with self.lock:
          self.box_dist[0,j] = min(self.box_dist[0,j], tmp1) 
          self.box_dist[1,j] = min(self.box_dist[1,j], tmp2) 
        sorted_indices = []
        if done == 0:
          sorted_indices = np.argsort(elm_dists)[nsort:]
        for i in sorted_indices:
          if max(self.box_dist[0,j], self.box_dist[1,j]) < elm_dists[i] - pad:
            done = 1
            break 
          dist = np.sqrt(
                         np.square(pos_elm[0,i] + X - self.box_pos[0,j]) 
                       + np.square(pos_elm[1,i] + Y - self.box_pos[1,j]) 
                       + np.square(pos_elm[2,i] + Z - self.box_pos[2,j]) 
                        )
          tmp1 = np.amin(np.where(fs[:,:,:,i]>0, dist, np.inf))
          tmp2 = np.amin(np.where(fs[:,:,:,i]<0, dist, np.inf))
          #with self.lock:
          self.box_dist[0,j] = min(self.box_dist[0,j], tmp1) 
          self.box_dist[1,j] = min(self.box_dist[1,j], tmp2) 
        search_time += timer.time()
      print("prebox time {:f}".format(sort_time))
      print("box time {:f}".format(search_time))

def plot_slice(grid, fname = None, zslice = False, time = 0., height = None):
  import matplotlib
  matplotlib.rc('font', size=8)
  import matplotlib.pyplot as plt
  import numpy as np
  nplot = 5

  image_y = 24
  if zslice:
    image_x = int(image_y * grid.shape[0] / grid.shape[1] + .5)
  else:
    image_x = nplot*int(image_y * grid.shape[0] / grid.shape[2] + .5)
  image_x = max( image_x,  image_y * 1050/1680 )
  image_x = min( image_x,  image_y * 1680/1050 )

  if zslice:
    fig = plt.figure(figsize=(image_x,image_y))
    ax1 = plt.subplot(1,1,1)
    plt.title('Z-normal slice @ t={:3.2f}'.format(time))
    ax1.imshow(grid.zslice.transpose(), origin = 'lower', 
      interpolation='bicubic', 
      vmin = 0., vmax = 1., 
      aspect = 'auto',
      extent=[grid.origin[0],grid.corner[0],grid.origin[1],grid.corner[1]])
    plt.ylabel('Y')
    plt.xlabel('X')
    #plt.colorbar()
  else:
    fig = plt.figure(figsize=(image_x,image_y))

    ax1 = plt.subplot(1,nplot,1)
    ax1.imshow(grid.yslice.transpose(), origin = 'lower', 
      interpolation='bicubic', 
      vmin = 0., vmax = 1., 
      aspect = 'auto',
      extent=[grid.origin[0],grid.corner[0],grid.origin[2],grid.corner[2]] )
    if height != None:
      ax1.plot([grid.origin[0], grid.corner[0]], [height, height], linestyle='dashed', linewidth=1.0, color='w')
      ax1.plot([grid.origin[0], grid.corner[0]], [-height, -height], linestyle='dashed', linewidth=1.0, color='w')
      plt.xlim([grid.origin[0], grid.corner[0]])
      plt.ylim([grid.origin[2], grid.corner[2]])
    plt.ylabel('Z')
    plt.yticks(np.linspace(grid.origin[2],grid.corner[2], 5))
    plt.xticks(np.linspace(grid.origin[0],grid.corner[0], 3))

    umax = np.max(np.max(grid.yuzslice), np.max(grid.yuzslice))
    umin = np.min(np.min(grid.yuzslice), np.min(grid.yuzslice))

    ax2 = plt.subplot(1,nplot,2)
    ax2.imshow(grid.yuzslice.transpose(), origin = 'lower', 
      interpolation='bicubic',
      vmin = umin, vmax = umax, 
      aspect = 'auto',
      extent=[grid.origin[0],grid.corner[0],grid.origin[2],grid.corner[2]] )
    plt.yticks(np.linspace(grid.origin[2],grid.corner[2], 5))
    plt.xticks(np.linspace(grid.origin[0],grid.corner[0], 3))

    ax3 = plt.subplot(1,nplot,3)
    ax3.imshow(grid.yuxslice.transpose(), origin = 'lower', 
      interpolation='bicubic',
      vmin = umin, vmax = umax, 
      aspect = 'auto',
      extent=[grid.origin[0],grid.corner[0],grid.origin[2],grid.corner[2]] )
    plt.title('Y-normal slice @ t={:3.2f}'.format(time))
    plt.xlabel('X')
    plt.yticks(np.linspace(grid.origin[2],grid.corner[2], 5))
    plt.xticks(np.linspace(grid.origin[0],grid.corner[0], 3))

    ax4 = plt.subplot(1,nplot,4)
    ax4.imshow((
                grid.yuzslice[2:-1,1:-2]
              - grid.yuzslice[0:-3,1:-2]
              - grid.yuxslice[1:-2,2:-1]
              + grid.yuxslice[1:-2,0:-3]).transpose()/(2.*grid.dx[0]), origin = 'lower', 
      interpolation='bicubic',
      aspect = 'auto',
      extent=[grid.origin[0],grid.corner[0],grid.origin[2],grid.corner[2]] )
    plt.yticks(np.linspace(grid.origin[2],grid.corner[2], 5))
    plt.xticks(np.linspace(grid.origin[0],grid.corner[0], 3))

    ax5 = plt.subplot(1,nplot,5)
    ax5.imshow(grid.ypslice.transpose(), origin = 'lower', 
      interpolation='bicubic',
      aspect = 'auto',
      extent=[grid.origin[0],grid.corner[0],grid.origin[2],grid.corner[2]] )
    plt.yticks(np.linspace(grid.origin[2],grid.corner[2], 5))
    plt.xticks(np.linspace(grid.origin[0],grid.corner[0], 3))
    """
    ax6 = plt.subplot(1,nplot,6)
    ax6.imshow((
                grid.ypslice[1:-2,2:-1]
              + grid.ypslice[1:-2,0:-3]
              + grid.ypslice[2:-1,1:-2]
              + grid.ypslice[0:-3,1:-2]
            - 4*grid.ypslice[1:-2,1:-2]).transpose(), origin = 'lower', 
      interpolation='bicubic',
      aspect = 'auto',
      extent=[grid.origin[0],grid.corner[0],grid.origin[2],grid.corner[2]] )
    plt.yticks(np.linspace(grid.origin[2],grid.corner[2], 5))
    plt.xticks(np.linspace(grid.origin[0],grid.corner[0], 3))
    """

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

def plot_prof(grid, fname = None, line = None):
  import matplotlib.pyplot as plt
  import numpy as np

  fig = plt.figure()
  ax1 = plt.subplot(1,1,1)
  plt.title('Density profile')
  zs = np.linspace(grid.origin[2], grid.corner[2], grid.shape[2], endpoint = False)
  plt.plot(zs , grid.f_xy / (grid.shape[0]*grid.shape[1]) )
  if line != None:
    plt.plot(zs, line*zs + .5)
  plt.ylim([0,1])
  plt.xlabel('z')
  plt.ylabel('<f>_{x,y}')
  if fname != None:
    plt.savefig(fname)

def plot_dim(grid, fname = None):
  import matplotlib.pyplot as plt
  import numpy as np
  plt.figure()
  ax1 = plt.subplot(1,1,1)
  grid.box_dist *= 1./(np.linalg.norm(grid.corner - grid.origin))
  n, bins, patches = ax1.hist(np.maximum(grid.box_dist[0,:], grid.box_dist[1,:]),
                              bins=grid.nbox,
                              cumulative=True,
                              log=True,
                              normed = True,
                              histtype = 'stepfilled'
                             )
  xs = np.array([bins[0]/2., 1./3.])
  for i in range(-10, 11):
    ax1.plot(xs, xs*(2.**i), 'k--')
#  for i in range(-10, 11):
#    ax1.plot(xs, np.square(xs)*(2.**i), 'r--')
#  for i in range(-10, 11):
#    ax1.plot(xs, np.multiply(np.square(xs), xs)*(2.**i), 'g--')
  #plt.xlim([bins[0]/2., 1.])
  plt.xlim([0.001, 1.])
  #plt.ylim([n[0]/2., 1.])
  plt.ylim([0.1, 1.])
  plt.vlines(1./3., 1./grid.shape[0], 1.)
  plt.xscale('log')
  plt.yscale('log')

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
  modes = np.zeros((modes_x.size, modes_y.size), order='F')
  for i in range(modes_x.size):
    for j in range(modes_y.size):
      modes[i,j] = np.sqrt(abs(modes_x[i])*abs(modes_x[i]) + abs(modes_y[j]) * abs(modes_y[j]))
  plt.xlim([modes_y[1]/1.5, np.max(modes)*1.5])

  for zpos in slices:
    z = int(zpos * grid.shape[2])

    # compute Taylor microscale
    spectrum = np.square(np.abs(np.fft.rfft2(grid.zslice[:,:])) / (grid.shape[0]*grid.shape[1]))
    '''
    for i in range(modes_x.size):
     for j in range(modes_y.size):
      if spectrum[i,j] > 1e-10:
        print("Modes {:f} and {:f} has amplitude {:e}".format(modes_x[i], modes_y[j], spectrum[i,j]))
    '''
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
  
  xs = np.sort(modes.ravel())[1:]
  ys = xs**(-5./3.) 
  for i in range(9):
    ax1.plot(xs, 10**(1.-2.*i) * ys, 'k--')
  ax1.plot(xs, .25 * Atwood * g * np.square(0.1 / xs), 'y-')
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

def mixing_zone(grid, thresh = .05):
  import numpy as np
  from my_utils import find_root
  from tictoc import tic, toc

  L = np.max(grid.corner[2]) - np.min(grid.origin[2])

  tic()
  f_xy = np.ones(grid.shape[2])
  for i in range(grid.shape[2]):
    f_xy[i] = grid.f_xy[i] / (grid.shape[0]*grid.shape[1]) 
  toc('mix_renorm')

  # Cabot's h
  tic()
  h = 0.
  for i in range(f_xy.shape[0]):
    if f_xy[i] < .5:
      h += 2*f_xy[i]
    else:
      h += 2*(1.-f_xy[i])
  h_cabot = (h * L / f_xy.shape[0])
  toc('mix_cabot')

  # visual h
  tic()
  zs = np.linspace(grid.origin[2], grid.corner[2], grid.shape[2], endpoint = False)
  toc('mix_linspace')
  tic()
  h_visual = ( find_root(zs, f_xy, y0 = thresh) 
             - find_root(zs, f_xy, y0 = 1-thresh)) / 2.
  toc('mix_visual')

  spread = max(int(h_cabot * f_xy.shape[0]/ (4.*L)), 1)
  p = np.polyfit(zs[  grid.shape[2]/2-spread:grid.shape[2]/2 + spread],
                 f_xy[grid.shape[2]/2-spread:grid.shape[2]/2 + spread], 1)
  h_fit = abs((1.)/(2. * p[0]))

  tic()
  X = float(grid.f_m/(h*grid.shape[0]*grid.shape[1]))
  lint = int((.5-h_visual/L)*grid.shape[2]+.5)
  hint = int((.5+h_visual/L)*grid.shape[2]+.5)
  T = float(np.sum(grid.ff_xy[lint:hint] / (f_xy[lint:hint] * (1-f_xy[lint:hint])))) 
  T = T * L / (2.*h_visual) / np.prod(grid.shape)
  Y = float(grid.f_total/(np.prod(grid.shape)))
  toc('mix_agg')

  return h_cabot, h_visual, h_fit, X, T, Y

def energy_budget(grid):
  import numpy as np
  from my_utils import find_root

  # Potential
  zs = np.linspace(grid.origin[2], grid.corner[2], grid.shape[2], endpoint = False)
  dV = np.prod(grid.dx)
  U = 0.
  for i in range(grid.shape[2]):
    U = U - grid.f_xy[i] * zs[i] * dV
  U0 = np.prod(grid.shape)/2. * dV * zs[int(grid.shape[2]*3./4.)]

  # Kinetic
  K = grid.v2 * dV/2.
  return Atwood*g*(U0 - U), K

