
def find_root(x, y, y0 = .5, desired_resolution = None):
  from scipy import interpolate
  import numpy as np
  if desired_resolution == None:
    desired_resolution = abs(x[1] - x[0])/8.
  i_low = 0
  i_high = x.shape[0]
  failsafe = 0
  while (i_high - i_low) > 32 and failsafe < 10:
    failsafe += 1
    i_1 = int(i_low + (i_high-i_low)/3)
    i_2 = int(i_low + (i_high-i_low)*2/3)
    if y[i_1] > y0*1.1:
      i_low = i_1
    if y[i_2] < y0*.9:
      i_high = i_2

  f = interpolate.interp1d(x[i_low:i_high], y[i_low:i_high], kind='cubic')
  x_low  = np.min(x[i_low:i_high])
  x_high = np.max(x[i_low:i_high])
  x_guess = (x_high + x_low)/2.
  while (x_high - x_low) > desired_resolution:
    fx = f(x_guess)
    if fx > y0:
      x_low = x_guess
    else:
      x_high = x_guess
    x_guess = (x_high + x_low)/2.
  return x_guess

""" Build Lagrange interpolation matrix """
def lagrange_matrix(A,B):
  import numpy as np
  M = np.zeros((B.size,A.size), order='F')
  for i in range(A.size):
    for j in range(B.size):
      M[j,i] =  1.
      for k in range(A.size):
        if k == i:
          continue
        M[j,i] = M[j,i] * (B[j] - A[k]) / (A[i] - A[k])
  return M

def transform_elements(f, p, trans, cart):
  from tictoc import tic, toc
  import numpy as np
  ninterp = trans.shape[0]
  norder = trans.shape[1]
  nelm = f.shape[1]

  # Apply the transformation
  f_tmp = np.zeros((norder**2*ninterp,nelm), order='F')
  f_tmp2 = np.zeros((norder*ninterp**2,nelm), order = 'F')
  f_trans = np.zeros((ninterp**3,nelm), order = 'F')

  # Transform to uniform grid
  # z-first
  tic()
  f_p = np.reshape(np.transpose(np.reshape(f, (norder**2, norder, nelm), order='F'), (1,0,2)), (norder, norder**2*nelm), order='F')
  f_tmp = np.reshape(np.transpose(np.reshape(trans.dot(f_p), (ninterp, norder**2, nelm), order='F'), (1,0,2)), (norder**2*ninterp, nelm), order='F')
  toc('trans_z')

  # then x
  tic()
  f_tmp2 = np.reshape(trans.dot(np.reshape(f_tmp, (norder, ninterp*norder*nelm), order='F')), (ninterp**2*norder,nelm), order='F')
  toc('trans_x')

  # then y
  tic()
  f_p =     np.reshape(np.transpose(np.reshape(f_tmp2,         (ninterp, norder, ninterp, nelm),  order='F'), (1,0,2,3)), (norder, ninterp**2*nelm), order='F')
  f_trans = np.reshape(np.transpose(np.reshape(trans.dot(f_p), (ninterp, ninterp, ninterp, nelm), order='F'), (1,0,2,3)), (ninterp**3, nelm),        order='F')
  toc('trans_y')

  # Transform positions to uniform grid
  tic()
  pos_tmp = np.zeros((ninterp, ninterp, ninterp, 3), order='F')
  pos_trans = np.zeros((ninterp**3, nelm, 3), order='F')
  for i in range(nelm):
    for j in range(ninterp):
      pos_tmp[:,j,:,1] = p[0,i,1] + cart[j]
      pos_tmp[j,:,:,0] = p[0,i,0] + cart[j]
      pos_tmp[:,:,j,2] = p[0,i,2] + cart[j]
    for j in range(3):
      pos_trans[:,i,j] = pos_tmp[:,:,:,j].flatten(order='F')
  return f_trans, pos_trans

