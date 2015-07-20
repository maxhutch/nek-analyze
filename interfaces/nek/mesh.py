
from interfaces.abstract import AbstractMesh
from interfaces.nek.sem import zwgll, dhat
from interfaces.nek.slice import DenseSlice
import numpy as np

class GeneralMesh(AbstractMesh):
  """ Most general Nek mesh; doesn't support much. """
  def __init__(self, reader, params):
    self.reader = reader
    self.norder = reader.norder
    self.nx     = reader.nx
    self.ny     = reader.ny
    self.nz     = reader.nz
    self.origin = np.array(params['root_mesh']) 
    self.corner = np.array(params['extent_mesh'])
    self.extent = self.corner - self.origin
    self.shape  = np.array(params['shape_mesh'])
    self.length = self.extent / self.shape
    self.fields = {}
    self.dealias = 1.
    self.zx, self.wx = zwgll(self.nx-1)
    self.zy, self.wy = zwgll(self.ny-1)
    self.zz, self.wz = zwgll(self.nz-1)

    return

  def load(self, pos, num):
    n, x, u, p, t = self.reader.get_elem(num,pos)
    self.nelm = int(n)
    nshp = (self.norder, self.norder, self.norder, self.nelm)
    self.fields['x'] = np.reshape(x[:,0,:], nshp, order = 'F')
    self.fields['y'] = np.reshape(x[:,1,:], nshp, order = 'F')
    self.fields['z'] = np.reshape(x[:,2,:], nshp, order = 'F')
    self.fields['u'] = np.reshape(u[:,0,:], nshp, order = 'F')
    self.fields['v'] = np.reshape(u[:,1,:], nshp, order = 'F')
    self.fields['w'] = np.reshape(u[:,2,:], nshp, order = 'F')
    self.fields['p'] = np.reshape(p       , nshp, order = 'F')
    self.fields['t'] = np.reshape(t       , nshp, order = 'F')
    self.root = np.zeros((3,self.nelm), order='F', dtype=int)
    for i in range(3):
      self.root[i,:] = (x[0,i,:] -self.origin[i]) / self.length[i]
    self.b3 = np.zeros(nshp)
    for i in range(self.nelm):
        lx = abs(self.fields['x'][-1,0,0,i] - self.fields['x'][0,0,0,i])
        gllx = lx * (self.zx+1.)/(2.)
        b1x  = self.wx * (lx / 2.)

        ly = abs(self.fields['y'][0,-1,0,i] - self.fields['y'][0,0,0,i])
        glly = ly * (self.zy+1.)/(2.)
        b1y  = self.wy * (ly / 2.)

        lz = abs(self.fields['z'][0,0,-1,i] - self.fields['z'][0,0,0,i])
        gllz = lz * (self.zz+1.)/(2.)
        b1z  = self.wz * (lz / 2.)

        b2xy = np.outer(b1y,b1z)
        self.b3[:,:,:,i] = np.reshape(np.outer(b1x, b2xy), (self.norder, self.norder, self.norder))

    return

  def fld(self, name):
    return self.fields[name]

  def dx(self, fld, axis):
    if isinstance(fld, str):
      fld = self.fld(fld)

    raise ValueError("GeneralMesh doesn't support differentiation")

  def int(self, fld, axis = (0,1,2,3)):
    if isinstance(fld, str):
      fld = self.fld(fld)

    # Note, this isn't quite right
    if len(axis) == 4:
      foo = fld * self.b3
      return np.add.reduce(foo, axis)
    raise ValueError("GeneralMesh doesn't support partial intergration")

  def max(self, fld, axis = (0,1,2,3)):
    if isinstance(fld, str):
      fld = self.fld(fld)
    return np.maximum.reduce(fld,axis)

  def min(self, fld, axis = (0,1,2,3)):
    if isinstance(fld, str):
      fld = self.fld(fld)
    return np.minimum.reduce(fld,axis)


class UniformMesh(GeneralMesh):
  """ Mesh of uniform elements """
  def __init__(self, reader, params):
    self.reader = reader
    self.norder = reader.norder
    self.origin = np.array(params['root_mesh']) 
    self.corner = np.array(params['extent_mesh'])
    self.extent = self.corner - self.origin
    self.shape  = np.array(params['shape_mesh'])
    self.length = self.extent / self.shape
    self.fields = {}
    self.dealias = 1.
    z, w = zwgll(self.norder-1)
    self.gll = self.length[0] * (z+1.)/(2.)
    self.b1  = w * (self.length[0] / 2.)
    self.b2  = np.outer(self.b1, self.b1)
    #self.b2[:,0]     = 0.5*self.b2[:,0]
    #self.b2[:,-1]    = 0.5*self.b2[:,-1]
    #self.b2[0,:]     = 0.5*self.b2[0,:]
    #self.b2[-1,:]    = 0.5*self.b2[-1,:]
    self.b2z = np.tile(self.b2, (self.norder,1,1)).transpose()
    #self.b2z[:,:,0]  = 0.5*self.b2z[:,:,0]
    #self.b2z[:,:,-1] = 0.5*self.b2z[:,:,-1]
    self.b3  = np.reshape(np.outer(self.b1,self.b2),
                          (self.norder,self.norder, self.norder))
    self.d1  = dhat(self.gll)

    return

  def load(self, pos, num):
    n, x, u, p, t = self.reader.get_elem(num,pos)
    self.nelm = int(n)
    nshp = (self.norder, self.norder, self.norder, self.nelm)
    self.fields['x'] = np.reshape(x[:,0,:], nshp, order = 'F')
    self.fields['y'] = np.reshape(x[:,1,:], nshp, order = 'F')
    self.fields['z'] = np.reshape(x[:,2,:], nshp, order = 'F')
    self.fields['u'] = np.reshape(u[:,0,:], nshp, order = 'F')
    self.fields['v'] = np.reshape(u[:,1,:], nshp, order = 'F')
    self.fields['w'] = np.reshape(u[:,2,:], nshp, order = 'F')
    self.fields['p'] = np.reshape(p       , nshp, order = 'F')
    self.fields['t'] = np.reshape(t       , nshp, order = 'F')
    self.root = np.zeros((3,self.nelm), order='F', dtype=int)
    for i in range(3):
      self.root[i,:] = (x[0,i,:] -self.origin[i]) / self.length[i]
    return

  def dx(self, fld, axis):
    if isinstance(fld, str):
      fld = self.fld(fld)

    res = np.tensordot(self.d1, fld, axes=([1,axis]))
    if axis == 1:
      res = res.transpose([1,0,2,3])
    elif axis == 2:
      res = res.transpose([1,2,0,3])
    return res

  def int(self, fld, axis = (0,1,2,3)):
    if isinstance(fld, str):
      fld = self.fld(fld)

    # Note, this isn't quite right
    if len(axis) == 4:
      foo = fld * np.tile(self.b3, (self.nelm,1,1,1)).transpose()
      return np.add.reduce(foo, axis)
    if len(axis) == 2 and axis[0] == 0 and axis[1] == 1:
      foo = fld*np.tile(self.b2z.transpose(), (self.nelm,1,1,1)).transpose()
      return np.add.reduce(foo, axis)

  def slice(self, fld, intercept, axis, op = None):
    if isinstance(fld, str):
        fld = self.fld(fld) 
 
    # init meta-data
    full_shape = self.shape * (self.norder-1) + 1
    slice_shape = []
    root = []
    cept = []
    root2 = []
    #p = ['x', 'y', 'z']
    for i in range(3):
      if not i in axis:
        root.append(i)
        slice_shape.append(full_shape[i]-1)
      else:
        root2.append(i)
        cept.append(int((intercept[i] -self.origin[i]) / self.length[i]))

    # Init res
    res = DenseSlice(slice_shape, op)
 
    if op != None:
      if op == 'int':
        local = self.int(fld[:,:,:,:], axis)
        local = local[...,:-1,:]
      else:
        local = op.reduce(fld[:-1,:-1,:-1,:], axis)

      sls = [[(self.norder - 1) * self.root[root[j],i] for j in range(len(root))] for i in range(self.nelm)]
      for i in range(self.nelm):
          res.add(sls[i], local[...,i])
 
    else:
      local = fld[:-1,:-1,:-1,:]
      for i in range(self.nelm):
        here = all([self.root[root2[j],i] == cept[j] for j in range(len(cept))])
        if not here: 
          continue
        sl = tuple([np.s_[0] if ax in axis else np.s_[:] for ax in range(3)] + [np.s_[i]])
        foo = local[sl]
        starti = [(self.norder-1)*self.root[root[j],i] for j in range(len(root))]
        res.add(starti, foo)
    
    return res

class GeneralMesh(AbstractMesh):
  """ Most general Nek mesh; doesn't support much. """
  def __init__(self, reader, params):
    self.reader = reader
    self.norder = reader.norder
    self.nx     = reader.nx
    self.ny     = reader.ny
    self.nz     = reader.nz
    self.origin = np.array(params['root_mesh']) 
    self.corner = np.array(params['extent_mesh'])
    self.extent = self.corner - self.origin
    self.shape  = np.array(params['shape_mesh'])
    self.length = self.extent / self.shape
    self.fields = {}
    self.dealias = 1.
    self.zx, self.wx = zwgll(self.nx-1)
    self.zy, self.wy = zwgll(self.ny-1)
    self.zz, self.wz = zwgll(self.nz-1)

    return

  def load(self, pos, num):
    n, x, u, p, t = self.reader.get_elem(num,pos)
    self.nelm = int(n)
    nshp = (self.norder, self.norder, self.norder, self.nelm)
    self.fields['x'] = np.reshape(x[:,0,:], nshp, order = 'F')
    self.fields['y'] = np.reshape(x[:,1,:], nshp, order = 'F')
    self.fields['z'] = np.reshape(x[:,2,:], nshp, order = 'F')
    self.fields['u'] = np.reshape(u[:,0,:], nshp, order = 'F')
    self.fields['v'] = np.reshape(u[:,1,:], nshp, order = 'F')
    self.fields['w'] = np.reshape(u[:,2,:], nshp, order = 'F')
    self.fields['p'] = np.reshape(p       , nshp, order = 'F')
    self.fields['t'] = np.reshape(t       , nshp, order = 'F')
    self.root = np.zeros((3,self.nelm), order='F', dtype=int)
    for i in range(3):
      self.root[i,:] = (x[0,i,:] -self.origin[i]) / self.length[i]
    self.b3 = np.zeros(nshp)
    for i in range(self.nelm):
        lx = abs(self.fields['x'][-1,0,0,i] - self.fields['x'][0,0,0,i])
        gllx = lx * (self.zx+1.)/(2.)
        b1x  = self.wx * (lx / 2.)

        ly = abs(self.fields['y'][0,-1,0,i] - self.fields['y'][0,0,0,i])
        glly = ly * (self.zy+1.)/(2.)
        b1y  = self.wy * (ly / 2.)

        lz = abs(self.fields['z'][0,0,-1,i] - self.fields['z'][0,0,0,i])
        gllz = lz * (self.zz+1.)/(2.)
        b1z  = self.wz * (lz / 2.)

        b2xy = np.outer(b1x,b1y)
        self.b3[:,:,:,i] = np.reshape(np.outer(b1z, b2xy), (self.norder, self.norder, self.norder))

    return

  def fld(self, name):
    return self.fields[name]

  def dx(self, fld, axis):
    if isinstance(fld, str):
      fld = self.fld(fld)

    raise ValueError("GeneralMesh doesn't support differentiation")

  def int(self, fld, axis = (0,1,2,3)):
    if isinstance(fld, str):
      fld = self.fld(fld)

    # Note, this isn't quite right
    if len(axis) == 4:
      foo = fld * self.b3
      return np.add.reduce(foo, axis)
    raise ValueError("GeneralMesh doesn't support partial intergration")

  def max(self, fld, axis = (0,1,2,3)):
    if isinstance(fld, str):
      fld = self.fld(fld)
    return np.maximum.reduce(fld,axis)

  def min(self, fld, axis = (0,1,2,3)):
    if isinstance(fld, str):
      fld = self.fld(fld)
    return np.minimum.reduce(fld,axis)

