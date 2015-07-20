from interfaces.abstract import AbstractSlice
import numpy as np

class DenseSlice(AbstractSlice):
  """ Uninspired dense slice """

  def __init__(self, shape, op=None):
    self.shape = shape
    self.op = op
    if self.op is 'int':
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
    if self.op is None:
        self.sl = self.sl + sl2.to_array()
    else:
        self.sl = self.op(self.sl, sl2.to_array())

  def add(self, pos, data):
    block = data.shape
    idx = tuple([np.s_[pos[j]:pos[j]+block[j]] for j in range(len(pos))])
    if self.op is None:
        self.sl[idx] += data
    else:
        self.sl[idx] = self.op(self.sl[idx], data)
