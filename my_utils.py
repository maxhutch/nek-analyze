
def find_root(x, y, y0 = .5, desired_resolution = None):
  from scipy import interpolate
  import numpy as np
  if desired_resolution == None:
    desired_resolution = abs(x[1] - x[0])/64.
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
  M = np.zeros((B.size,A.size), order='F', dtype=np.float64)
  for i in range(A.size):
    for j in range(B.size):
      M[j,i] =  1.
      for k in range(A.size):
        if k == i:
          continue
        M[j,i] = M[j,i] * (B[j] - A[k]) / (A[i] - A[k])
  return M

def transform_field_elements(f, trans, cart):
  from tictoc import tic, toc
  import numpy as np
  import gc
#  f = args['f']
#  trans = args['trans']
#  cart = args['cart']
  ninterp = trans.shape[0]
  norder = trans.shape[1]
  nelm = f.shape[1]

  # Transform to uniform grid
  # z-first
  tic()
  f_p = np.reshape(np.transpose(np.reshape(f, (norder**2, norder, nelm), order='F'), (1,0,2)), (norder, norder**2*nelm), order='F')
  f_tmp = np.reshape(np.transpose(np.reshape(trans.dot(f_p), (ninterp, norder**2, nelm), order='F'), (1,0,2)), (norder, norder*ninterp*nelm), order='F')
  toc('trans_z')

  # then x
  tic()
  f_tmp2 = np.reshape(trans.dot(f_tmp), (ninterp, norder, ninterp,nelm), order='F')
  toc('trans_x')

  # then y
  tic()
  f_p =     np.reshape(np.transpose(f_tmp2, (1,0,2,3)), (norder, ninterp**2*nelm), order='F')
  f_trans = np.reshape(np.transpose(np.reshape(trans.dot(f_p), (ninterp, ninterp, ninterp, nelm), order='F'), (1,0,2,3)), (ninterp**3, nelm),        order='F')
  toc('trans_y')

  f_p = None; f_tmp2 = None; f_tmp = None
  gc.collect()

  return f_trans

from threading import Thread
class TransformFieldElements(Thread):
  def  __init__(self, f, trans, cart):
    Thread.__init__(self)
    self.f = f
    self.trans = trans
    self.cart = cart

  def run(self):
    self.f_trans = transform_field_elements(self.f, self.trans, self.cart)

def transform_position_elements(p, trans, cart):
  from tictoc import tic, toc
  import numpy as np
  ninterp = trans.shape[0]
  norder = trans.shape[1]
  nelm = p.shape[1]

  # Transform positions to uniform grid
  tic()
  pos_tmp = np.zeros((ninterp, ninterp, ninterp), order='F', dtype=np.float64)
  pos_trans = np.zeros((ninterp**3, nelm, 3),     order='F', dtype=np.float64)
  block_x = np.zeros((ninterp,ninterp,ninterp),   order='F', dtype=np.float64)
  block_y = np.zeros((ninterp,ninterp,ninterp),   order='F', dtype=np.float64)
  block_z = np.zeros((ninterp,ninterp,ninterp),   order='F', dtype=np.float64)
  for j in range(ninterp):
    block_x[j,:,:] = cart[j]
    block_y[:,j,:] = cart[j]
    block_z[:,:,j] = cart[j]
  for i in range(nelm):
    pos_tmp[:,:,:] = p[0,i,0] + block_x
    pos_trans[:,i,0] = pos_tmp[:,:,:].flatten(order='F')
  for i in range(nelm):
    pos_tmp[:,:,:] = p[0,i,1] + block_y
    pos_trans[:,i,1] = pos_tmp[:,:,:].flatten(order='F')
  for i in range(nelm):
    pos_tmp[:,:,:] = p[0,i,2] + block_z
    pos_trans[:,i,2] = pos_tmp[:,:,:].flatten(order='F')
  toc('trans_pos')
  return pos_trans

from threading import Thread
class TransformPositionElements(Thread):
  def  __init__(self, p, trans, cart):
    Thread.__init__(self)
    self.p = p
    self.trans = trans
    self.cart = cart

  def run(self):
    self.p_trans = transform_position_elements(self.p, self.trans, self.cart)


def compute_index(root, shape):
  import numpy as np
  if np.all(shape == 0):
    return 0
  power = np.prod(shape) / 8
  frac_root = np.divide(root.astype(np.float64), shape)
  ind = 0
  if frac_root[0] >= .5:
    ind = ind + 1
  if frac_root[1] >= .5:
    ind = ind + 2
  if frac_root[2] >= .5:
    ind = ind + 4
  shape = np.array(shape / 2, dtype = int)
  return ind * power + compute_index(np.mod(root,shape), shape)

