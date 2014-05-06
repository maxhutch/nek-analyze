
class NekFile():
  def __init__(self, fname):
    import struct
    self.x_file = open(fname, 'rb')
    header = str(self.x_file.read(132))
    htoks = header.split()
    self.nelm = int(htoks[5])
    self.norder = int(htoks[2])
    self.time = float(htoks[7])
    print("Read {:d} elements of order {:d} at time {:f}".format(self.nelm, self.norder, self.time))
    #''' Assume isotropic elements '''
    self.ntot = self.nelm * self.norder**3
    #''' Check the test float '''
    test = struct.unpack('f', self.x_file.read(4))
    byteswap = abs(test[0] - 6.543210029) > 0.00001
    if byteswap:
      print("  * swapping bytes")
      self.ty = '>f4'
    else:
      self.ty = 'f4'
    self.current_elm = 0
    self.u_file = open(fname, 'rb')
    self.t_file = open(fname, 'rb')
    self.x_file.seek(136+self.nelm*4,               0) 
    self.u_file.seek(136+self.nelm*4+3*self.ntot*4, 0) 
    self.t_file.seek(136+self.nelm*4+7*self.ntot*4, 0) 

  def close(self):
    self.x_file.close()
    self.u_file.close()
    self.t_file.close()
    return

  def get_elem(self, num):
    import numpy as np
    num = min(num, self.nelm - self.current_elm)
    print("Returning {:d} elements".format(num))
    if num < 0:
      return 0, None, None, None
    x_raw = np.fromfile(self.x_file, dtype=self.ty, count = num*(self.norder**3)*3) 
    u_raw = np.fromfile(self.u_file, dtype=self.ty, count = num*(self.norder**3)*3) 
    t_raw = np.fromfile(self.t_file, dtype=self.ty, count = num*(self.norder**3)) 
    
    x = np.transpose(np.reshape(x_raw, (self.norder**3,3,num), order='F'), (0,2,1))
    u = np.transpose(np.reshape(u_raw, (self.norder**3,3,num), order='F'), (0,2,1))
    t =              np.reshape(t_raw, (self.norder**3,  num), order='F')

    self.current_elm += num
    return num, x, u, t

