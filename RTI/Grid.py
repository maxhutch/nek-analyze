
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
    self.yind       = int(self.shape[1]/2. + .5)
    self.yslice     = np.zeros((self.shape[0], self.shape[2]), order='F')
    self.ypslice    = np.zeros((self.shape[0], self.shape[2]), order='F')
    self.yuzslice   = np.zeros((self.shape[0], self.shape[2]), order='F')
    self.yuxslice   = np.zeros((self.shape[0], self.shape[2]), order='F')
    self.zind       = int(self.shape[2]/2. + .5)
    self.zslice     = np.zeros((self.shape[0], self.shape[1]), order='F')
    self.zsliceu    = np.zeros((self.shape[0], self.shape[1],3), order = 'F')
    self.dotzsliceu = np.zeros((self.shape[0], self.shape[1]), order = 'F')

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
    self.f_xy[:]         = 0
    self.ff_xy[:]        = 0 
    self.yslice[:,:]     = 0 
    self.ypslice[:,:]    = 0 
    self.yuzslice[:,:]   = 0
    self.yuxslice[:,:]   = 0
    self.zslice[:,:]     = 0
    self.zsliceu[:,:]    = 0
    self.dotzsliceu[:,:] = 0 

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

        self.ypslice[root[0]:root[0]+self.order, 
                     root[2]:root[2]+self.order] = p_tmp[:,yoff,:]

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

