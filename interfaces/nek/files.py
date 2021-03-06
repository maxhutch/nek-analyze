def nek_fname(name, frame = 1, proc = 0, io_files = 1):
  from math import log10
  from os.path import dirname, basename
  data_path = dirname(name)
  data_tag  = basename(name) 
  dir_width = int(log10(max(abs(io_files)-1,1)))+1
  if io_files > 0:
    fname = "{:s}{:0{width}d}.f{:05d}".format(name, proc, frame, width=dir_width)
  else:
    fname = "{:s}/A{:0{width}d}/{:s}{:0{width}d}.f{:05d}".format(data_path, proc, data_tag, proc, frame, width=dir_width)
  return fname


from interfaces.abstract import AbstractFileReader
class NekFile(AbstractFileReader):
  def __init__(self, f, pf = None, base = None):
    # Do we have another file to base this off of?
    if base != None:
      self.init2(base, fname)
      return

    import struct
    self.f = f
    self.pf = pf

    self.header = self.f.read(132)
    htoks = str(self.header).split()
    self.word_size = int(htoks[1])
    self.norder = int(htoks[2])
    self.nelm = int(htoks[5])
    self.time = float(htoks[7])
    if htoks[0] == "b'#max":
      self.padded = 8 * (2**20)
    else:
      self.padded = -1

    #print("Read {:d} elements of order {:d} at time {:f}".format(self.nelm, self.norder, self.time))

    #''' Assume isotropic elements '''
    self.ntot = self.nelm * self.norder**3

    #''' Check the test float '''
    self.test = self.f.read(4)
    self.test_tuple = struct.unpack('f', self.test)
    byteswap = abs(self.test_tuple[0] - 6.543210029) > 0.00001
    if byteswap:
      self.ty = '>f{:1d}'.format(self.word_size)
    else:
      self.ty = 'f{:1d}'.format(self.word_size)
    self.bformat = None

    # Test opening the file
    self.current_elm = 0
    self.seek(0,0)

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
    self.word_size = base.word_size

    # Test opening the file
    self.current_elm = 0
    self.u_file = open(fname, 'wb')
    self.p_file = open(fname, 'wb')
    self.t_file = open(fname, 'wb')
    self.seek(0,0)

  def close(self):
    """Close file pointers and point handles to None."""
    return

  def seek(self, ielm, ifield, f = None):
    """Move file pointers to point to the ielm-th element of the ifield-th field"""

    # Seek to the right positions
    if self.padded >= 0:
      pad = self.padded + self.padded*(int((self.nelm*4 - 1)/self.padded) + 1)
    else:
      pad = 136 + self.nelm*4

    #        offset -v                 header and map -v        field -v
    if f is None:
      self.f.seek(ielm*self.word_size*self.norder**3 + pad + ifield*self.ntot*self.word_size, 0) 
    else:
      f.seek(ielm*self.word_size*self.norder**3 + pad + ifield*self.ntot*self.word_size, 0) 

    return

  def get_elem(self, num = 1024, pos = -1):
    """Read sequential elements."""
    import numpy as np
    if pos < 0:
      pos = self.current_elm
    else:
      self.current_elm = pos

    numl = min(num, self.nelm - pos)
    if numl <= 0:
      return 0, None, None, None

    if self.pf is None or self.pf.name == self.f.name:
      self.seek(pos*3, 0)
      x_raw = np.fromfile(self.f, dtype=self.ty, count = numl*(self.norder**3)*3).astype(np.float64) 
      self.seek(pos*3, 3)
      u_raw = np.fromfile(self.f, dtype=self.ty, count = numl*(self.norder**3)*3).astype(np.float64) 
      self.seek(pos, 6)
      p_raw = np.fromfile(self.f, dtype=self.ty, count = numl*(self.norder**3)).astype(np.float64) 
      self.seek(pos, 7)
      t_raw = np.fromfile(self.f, dtype=self.ty, count = numl*(self.norder**3)).astype(np.float64) 
    else:
      if pos == 0:
        print("Using pos file")
      self.seek(pos*3, 0, self.pf)
      x_raw = np.fromfile(self.pf, dtype=self.ty, count = numl*(self.norder**3)*3).astype(np.float64) 
      self.seek(pos*3, 0)
      u_raw = np.fromfile(self.f, dtype=self.ty, count = numl*(self.norder**3)*3).astype(np.float64) 
      self.seek(pos, 3)
      p_raw = np.fromfile(self.f, dtype=self.ty, count = numl*(self.norder**3)).astype(np.float64) 
      self.seek(pos, 4)
      t_raw = np.fromfile(self.f, dtype=self.ty, count = numl*(self.norder**3)).astype(np.float64) 
   
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

class NekFld(AbstractFileReader):
  def __init__(self, f, pf = None, base = None):
    # Do we have another file to base this off of?
    if base != None:
      self.init2(base, fname)
      return

    import struct
    self.f = f
    self.pf = pf

    self.header = self.f.read(80)
    htoks = str(self.header).split()
    self.word_size = 4
    self.norder = int(htoks[2])
    self.nx = int(htoks[2])
    self.ny = int(htoks[3])
    self.nz = int(htoks[4])
    self.nelm = int(htoks[1])
    self.time = float(htoks[5])
    if htoks[0] == "b'#max":
      self.padded = 8 * (2**20)
    else:
      self.padded = -1

    #print("Read {:d} elements of order {:d} at time {:f}".format(self.nelm, self.norder, self.time))

    #''' Assume isotropic elements '''
    self.ntot = self.nelm * self.norder**3

    #''' Check the test float '''
    self.test = self.f.read(4)
    self.test_tuple = struct.unpack('f', self.test)
    byteswap = abs(self.test_tuple[0] - 6.543210029) > 0.00001
    if byteswap:
      self.ty = '>f{:1d}'.format(self.word_size)
    else:
      self.ty = 'f{:1d}'.format(self.word_size)
    self.bformat = None

    # Test opening the file
    self.current_elm = 0
    self.seek(0,0)

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
    self.word_size = base.word_size

    # Test opening the file
    self.current_elm = 0
    self.u_file = open(fname, 'wb')
    self.p_file = open(fname, 'wb')
    self.t_file = open(fname, 'wb')
    self.seek(0,0)

  def close(self):
    """Close file pointers and point handles to None."""
    return

  def seek(self, ielm, ifield, f = None):
    """Move file pointers to point to the ielm-th element of the ifield-th field"""

    # Seek to the right positions
    if self.padded >= 0:
      pad = self.padded #+ self.padded*(int((self.nelm*4 - 1)/self.padded) + 1)
    else:
      pad = 84 #+ self.nelm*4

    #        offset -v                 header and map -v        field -v
    if f is None:
      self.f.seek(ielm*self.word_size*self.norder**3*8 + pad + ifield*self.ntot*self.word_size, 0) 
    else:
      f.seek(ielm*self.word_size*self.norder**3*8 + pad + ifield*self.ntot*self.word_size, 0) 

    return

  def get_elem(self, num = 1024, pos = -1):
    """Read sequential elements."""
    import numpy as np
    if pos < 0:
      pos = self.current_elm
    else:
      self.current_elm = pos

    numl = min(num, self.nelm - pos)
    if numl <= 0:
      return 0, None, None, None

    if self.pf is None:
      self.seek(pos, 0)
      x_raw = np.fromfile(self.f, dtype=self.ty, count = numl*(self.norder**3)*8).astype(np.float64) 
    else:
      self.seek(pos*3, 0, self.pf)
      x_raw = np.fromfile(self.pf, dtype=self.ty, count = numl*(self.norder**3)*3).astype(np.float64) 
      self.seek(pos*3, 0)
      u_raw = np.fromfile(self.f, dtype=self.ty, count = numl*(self.norder**3)*3).astype(np.float64) 
      self.seek(pos, 3)
      p_raw = np.fromfile(self.f, dtype=self.ty, count = numl*(self.norder**3)).astype(np.float64) 
      self.seek(pos, 4)
      t_raw = np.fromfile(self.f, dtype=self.ty, count = numl*(self.norder**3)).astype(np.float64) 
  
    x = np.zeros( (self.norder**3,3,numl) )
    u = np.zeros( (self.norder**3,3,numl) )
    raw_reshape = np.reshape(x_raw, (self.norder**3, 8, numl), order='F')
    x[:,0,:] = raw_reshape[:,0,:] 
    x[:,1,:] = raw_reshape[:,1,:]
    x[:,2,:] = raw_reshape[:,2,:]
    u[:,0,:] = raw_reshape[:,3,:] 
    u[:,1,:] = raw_reshape[:,4,:]
    u[:,2,:] = raw_reshape[:,5,:]
    p = raw_reshape[:,6,:] 
    t = raw_reshape[:,7,:] 

    self.current_elm += numl

    return numl, x, u, p, t

 


class DefaultFileReader(NekFile):
  pass 
