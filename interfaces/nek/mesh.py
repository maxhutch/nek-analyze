
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
    self.gll = (z+1.)/(2.*self.length[0])
    self.b1  = w * (self.length[0] / 2.)
    self.b2  = np.outer(self.b1, self.b1)
    self.b3  = np.reshape(np.outer(self.b1,self.b2),
                          (self.norder,self.norder, self.norder))
    self.d1  = dhat(self.gll)

    return

  def load(self, pos, num):
    n, x, u, p, t = self.reader.get_elem(num,pos)
    self.nelm = n
    nshp = (self.norder, self.norder, self.norder, self.nelm)
    self.fields['x'] = np.reshape(x[:,0,:], nshp)
    self.fields['y'] = np.reshape(x[:,1,:], nshp)
    self.fields['z'] = np.reshape(x[:,2,:], nshp)
    self.fields['u'] = np.reshape(u[:,0,:], nshp)
    self.fields['v'] = np.reshape(u[:,1,:], nshp)
    self.fields['w'] = np.reshape(u[:,2,:], nshp)
    self.fields['p'] = np.reshape(p       , nshp)
    self.fields['t'] = np.reshape(t       , nshp)
    return

  def fld(self, name):
    return self.fields[name]

  def dx(self, fld, axis):
    if isinstance(fld, str):
      fld = self.fld(fld)

    res = np.tensordot(self.d1, fld, axes=([1,axis]))

    return res

  def int(self, fld, axis):
    if isinstance(fld, str):
      fld = self.fld(fld)
    if not isinstance(axis, list):
      axis = [axis]

    # Note, this isn't quite right
    return np.add.reduce(fld * self.b3, axis)

  def max(self, fld, axis = (0,1,2,3)):
    if isinstance(fld, str):
      fld = self.fld(fld)
    return np.maximum.reduce(fld,axis)

  def min(self, fld, axis = (0,1,2,3)):
    if isinstance(fld, str):
      fld = self.fld(fld)
    return np.minimum.reduce(fld,axis)

