def from_nek(fname):
  import struct
  import numpy as np
  with open(fname,'rb') as f:
    #''' The header is 132 bytes long '''
    header = str(f.read(132))
    htoks = header.split()
    nelm = int(htoks[5])
    norder = int(htoks[2])
    time = float(htoks[7])
    print("Read {:d} elements of order {:d} at time {:f}".format(nelm, norder, time))
    #''' Assume isotropic elements '''
    ntot = nelm * norder * norder * norder
    #''' Check the test float '''
    test = struct.unpack('f', f.read(4))
    byteswap = abs(test[0] - 6.543210029) > 0.00001
    if byteswap:
      print("  * swapping bytes")
      ty = '>f4'
    else:
      ty = 'f4'
    #''' 4 byptes per element for an unused map '''
    element_map = f.read(nelm*4)
    #''' 4*3 bytes per basis function for position '''
    xyz  = np.fromfile(f, dtype=ty, count=ntot*3)
    #''' 4*3 bytes per basis function for velocity '''
    u    = np.fromfile(f, dtype=ty, count=norder*norder*norder*nelm*3)
    #''' 4 bytes per basis function for pressure '''
    p    = np.fromfile(f, dtype=ty, count=norder*norder*norder*nelm)
    #''' 4 bytes per basis function for temperature '''
    t_in = np.fromfile(f, dtype=ty, count=norder*norder*norder*nelm)

  #''' Reshape vector data '''
  pos = np.transpose(np.reshape(xyz, (norder*norder*norder,3,nelm), order='F'), (0,2,1))
  vel = np.transpose(np.reshape(u, (norder*norder*norder,3,nelm), order='F'), (0,2,1))
  #''' Reshape scaler data '''
  t = np.reshape(t_in, (norder*norder*norder,nelm), order='F')
  return pos, vel, t, time, norder

