
from interfaces.abstract import AbstractMesh
from interfaces.nek.sem import zwgll, dhat
import numpy as np

class UniformMesh(AbstractMesh):

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

  def fld(self, name):
    return self.fields[name]

  def dx(self, fld, axis):
    if isinstance(fld, str):
      fld = self.fld(fld)

    res = np.tensordot(self.d1, fld, axes=([1,axis]))
    if axis == 1:
      res = res.transpose([1,0,2,3])
    elif axis == 2:
      res = res.transpose([2,1,0,3])
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


  def max(self, fld, axis = (0,1,2,3)):
    if isinstance(fld, str):
      fld = self.fld(fld)
    return np.maximum.reduce(fld,axis)

  def min(self, fld, axis = (0,1,2,3)):
    if isinstance(fld, str):
      fld = self.fld(fld)
    return np.minimum.reduce(fld,axis)

  def slice(self, fld, intercept, axis, op = None):
    if isinstance(fld, str):
      fld = self.fld(fld)
    full_shape = self.shape * (self.norder-1) + 1
    slice_shape = []
    root = []
    cept = []
    root2 = []
    p = ['x', 'y', 'z']
    for i in range(3):
      if not i in axis:
        root.append(i)
        slice_shape.append(full_shape[i]-1)
      else:
        root2.append(i)
        cept.append(int((intercept[i] -self.origin[i]) / self.length[i]))

    slice = np.zeros(slice_shape)
    if op != None:
      if op == 'int':
        local = self.int(fld[:,:,:,:], axis)
        local = local[...,:-1,:]
      else:
        local = op.reduce(fld[:-1,:-1,:-1,:], axis)
      sls = [tuple([np.s_[(self.norder - 1) * self.root[root[j],i]:(self.norder - 1) * (self.root[root[j],i]+1)] for j in range(len(root))]) for i in range(self.nelm)]
      for i in range(self.nelm):
        slice[sls[i]] += local[...,i]
    else:
      local = fld[:-1,:-1,:-1,:]
      for i in range(self.nelm):
        here = all([self.root[root2[j],i] == cept[j] for j in range(len(cept))])
        if not here: 
          continue
        sl = tuple([np.s_[0] if ax in axis else np.s_[:] for ax in range(3)] + [np.s_[i]])
        foo = local[sl]
        starti = [7*self.root[root[j],i] for j in range(len(root))]
        endi = [x + self.norder - 1 for x in starti]
        sl = tuple([np.s_[starti[j]:endi[j]] for j in range(len(root))])
        slice[sl] += foo
    
    return slice
