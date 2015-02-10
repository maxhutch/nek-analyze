from abc import ABCMeta, abstractmethod

class AbstractFileReader(metaclass=ABCMeta):

  @abstractmethod
  def __init__(self, fname):
    pass

  @abstractmethod
  def close(self):
    pass

  @abstractmethod
  def get_elem(self, num, pos):
    pass

class AbstractMesh(metaclass=ABCMeta):
  @abstractmethod
  def __init__(self, reader):
    pass

  @abstractmethod
  def load(self, pos, num):
    pass

  @abstractmethod
  def fld(self, name):
    pass

  @abstractmethod
  def dx(self, fld, axis):
    pass

  @abstractmethod
  def int(self, fld, axis):
    pass

  @abstractmethod
  def max(self, fld, axis):
    pass

  @abstractmethod
  def min(self, fld, axis):
    pass

