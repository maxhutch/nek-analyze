from interfaces.abstract import AbstractSlice
import numpy as np

class DenseSlice(AbstractSlice):
  """ Uninspired dense slice """

  def __init__(self, shape, op=None):
    self.shape = shape
    self.op = op
    if self.op is 'int' or self.op is None:
        self.op = np.add
    if self.op is np.maximum:
        self.sl = np.zeros(self.shape) + np.finfo(np.float64).min
    elif self.op is np.minimum:
        self.sl = np.zeros(self.shape) + np.finfo(np.float64).max
    else:
        self.sl = np.zeros(self.shape)
    
  def to_array(self):
    return self.sl

  def merge(self, sl2):
    self.sl = self.op(self.sl, sl2.to_array())

  def add(self, pos, data):
    block = data.shape
    idx = tuple([np.s_[pos[j]:pos[j]+block[j]] for j in range(len(pos))])
    self.sl[idx] = self.op(self.sl[idx], data)


class SparseSlice(AbstractSlice):

  def __init__(self, shape, op=None):
    self.shape = shape
    self.op = op
    if self.op is 'int' or self.op is None:
        self.op = np.add
    self.patches = {}
    
  def to_array(self):
    if self.op is np.maximum:
        res = np.zeros(self.shape) + np.finfo(np.float64).min
    elif self.op is np.minimum:
        res = np.zeros(self.shape) + np.finfo(np.float64).max
    else:
        res = np.zeros(self.shape)

    for pos,patch in self.patches.items():
        shp = patch.shape
        idx = tuple([np.s_[pos[j]:pos[j]+shp[j]] for j in range(len(pos))])
        res[idx] = self.op(res[idx], patch)

    return res

  def merge(self, sl2):
    for pos,patch in sl2.patches.items():
      self.add(pos, patch)

  def add(self, pos, data):
    key = tuple(pos)
    if key in self.patches:
      self.patches[key] = self.op(self.patches[key], data)
    else:
      self.patches[key] = np.copy(data)

