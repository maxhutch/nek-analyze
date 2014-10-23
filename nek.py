
class NekFile():
  def __init__(self, fname, base = None):
    # Do we have another file to base this off of?
    if base != None:
      self.init2(base, fname)
      return

    import struct
    self.fname = fname
    self.x_file = open(fname, 'rb')
    self.header = self.x_file.read(132)
    htoks = str(self.header).split()
    self.nelm = int(htoks[5])
    self.norder = int(htoks[2])
    self.time = float(htoks[7])
    if htoks[0] == "b'#max":
      self.padded = 8 * (2**20)
    else:
      self.padded = -1

    #print("Read {:d} elements of order {:d} at time {:f}".format(self.nelm, self.norder, self.time))

    #''' Assume isotropic elements '''
    self.ntot = self.nelm * self.norder**3

    #''' Check the test float '''
    self.test = self.x_file.read(4)
    self.test_tuple = struct.unpack('f', self.test)
    byteswap = abs(self.test_tuple[0] - 6.543210029) > 0.00001
    if byteswap:
      print("  * swapping bytes")
      self.ty = '>f4'
    else:
      self.ty = 'f4'
    self.bformat = None

    # Test opening the file
    self.current_elm = 0
    self.u_file = open(fname, 'rb')
    self.p_file = open(fname, 'rb')
    self.t_file = open(fname, 'rb')
    self.seek(0)
    # Then close it
    self.close()

  def init2(self, base, fname):
    # copy state and write header
    self.fname = fname
    self.x_file = open(fname, 'wb')
    self.header = base.header
    self.x_file.write(self.header)
    self.nelm = base.nelm
    self.norder = base.norder
    self.time = base.time
    self.padded = base.padded
    self.ntot = base.ntot
    self.test = base.test
    self.x_file.write(self.test)
    self.ty = base.ty
    self.bformat = base.bformat

    # Test opening the file
    self.current_elm = 0
    self.u_file = open(fname, 'wb')
    self.p_file = open(fname, 'wb')
    self.t_file = open(fname, 'wb')
    self.seek(0)
    # Then close it
    self.close()

  def close(self):
    """Close file pointers and point handles to None."""
    if self.x_file != None:
      self.x_file.close(); self.x_file = None
      self.u_file.close(); self.u_file = None
      self.p_file.close(); self.p_file = None
      self.t_file.close(); self.t_file = None
    return

  def seek(self, ielm, readable = False, writable = False):
    """Move file pointers to point to the ielm-th element and file mode."""

    # Check if file is byte-writable
    if writable and (self.x_file == None or not self.x_file.writable()):
      # [re]open as byte-writable
      if self.x_file != None and not self.x_file.closed:
        self.x_file.close()
        self.u_file.close()
        self.p_file.close()
        self.t_file.close()
      self.x_file = open(self.fname, 'wb')
      self.u_file = open(self.fname, 'wb')
      self.p_file = open(self.fname, 'wb')
      self.t_file = open(self.fname, 'wb')
    # Check if file is byte-readable
    if readable and (self.x_file == None or not self.x_file.readable()):
      # [re]open as byte-readable
      if self.x_file != None and not self.x_file.closed:
        self.x_file.close()
        self.u_file.close()
        self.p_file.close()
        self.t_file.close()
      self.x_file = open(self.fname, 'rb')
      self.u_file = open(self.fname, 'rb')
      self.p_file = open(self.fname, 'rb')
      self.t_file = open(self.fname, 'rb')

    # Seek to the right positions
    if self.padded >= 0:
      pad = self.padded + self.padded*(int((self.nelm*4 - 1)/self.padded) + 1)
    else:
      pad = 136 + self.nelm*4

    #        offset -v          header and map -v        field -v
    self.x_file.seek(ielm*12*self.norder**3 + pad,                 0) 
    self.u_file.seek(ielm*12*self.norder**3 + pad + 3*self.ntot*4, 0) 
    self.p_file.seek(ielm*4 *self.norder**3 + pad + 6*self.ntot*4, 0) 
    self.t_file.seek(ielm*4 *self.norder**3 + pad + 7*self.ntot*4, 0) 

    return

  def get_elem(self, num = 1024, pos = -1):
    """Read sequential elements."""
    import numpy as np
    if pos < 0:
      pos = self.current_elm
      self.seek(self.current_elm, readable = True)
    else:
      self.current_elm = pos
      self.seek(pos, readable = True)

    numl = min(num, self.nelm - pos)
    if numl < 0:
      return 0, None, None, None

    x_raw = np.fromfile(self.x_file, dtype=self.ty, count = numl*(self.norder**3)*3).astype(np.float64) 
    u_raw = np.fromfile(self.u_file, dtype=self.ty, count = numl*(self.norder**3)*3).astype(np.float64) 
    p_raw = np.fromfile(self.p_file, dtype=self.ty, count = numl*(self.norder**3)).astype(np.float64) 
    t_raw = np.fromfile(self.t_file, dtype=self.ty, count = numl*(self.norder**3)).astype(np.float64) 
    
    x = np.reshape(x_raw, (self.norder**3,3,numl), order='F')
    u = np.reshape(u_raw, (self.norder**3,3,numl), order='F')
    p =              np.reshape(p_raw, (self.norder**3,  numl), order='F')
    t =              np.reshape(t_raw, (self.norder**3,  numl), order='F')

    self.current_elm += numl

    return numl, x, u, p, t

  def write(self, x, u, p, t, ielm = -1):
    """Write one element."""

    # create a binary formatter
    if self.bformat == None:
      import struct
      if self.ty == '>f4':
        fmt = '>{:d}f'.format(self.norder**3)
      else:
        fmt = '{:d}f'.format(self.norder**3)
      self.bformat = struct.Struct(fmt)

    # seek where we need to be
    if ielm >= 0:
      self.seek(ielm, writable=True)
    else:
      self.seek(self.current_elm, writable=True)

    # write a single element's worth of data using binary formatter
    import numpy as np
    self.x_file.write(self.bformat.pack(*(x[:,0,0].astype(np.float32).tolist())))
    self.x_file.write(self.bformat.pack(*(x[:,1,0].astype(np.float32).tolist())))
    self.x_file.write(self.bformat.pack(*(x[:,2,0].astype(np.float32).tolist())))

    self.u_file.write(self.bformat.pack(*(u[:,0,0].astype(np.float32).tolist())))
    self.u_file.write(self.bformat.pack(*(u[:,1,0].astype(np.float32).tolist())))
    self.u_file.write(self.bformat.pack(*(u[:,2,0].astype(np.float32).tolist())))

    self.p_file.write(self.bformat.pack(*(p[:,0].astype(np.float32).tolist())))
    self.t_file.write(self.bformat.pack(*(t[:,0].astype(np.float32).tolist())))

    return

