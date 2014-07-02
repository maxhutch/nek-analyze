
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
  ninterp = trans.shape[0]
  norder = trans.shape[1]
  nelm = f.shape[1]

  tic()
  # Transform to uniform grid
  # z-first
  f_p = np.reshape(np.transpose(np.reshape(f, (norder**2, norder, nelm), order='F'), (1,0,2)), (norder, norder**2*nelm), order='F')
  f_tmp = np.reshape(np.transpose(np.reshape(trans.dot(f_p), (ninterp, norder**2, nelm), order='F'), (1,0,2)), (norder, norder*ninterp*nelm), order='F')

  # then x
  f_tmp2 = np.reshape(trans.dot(f_tmp), (ninterp, norder, ninterp,nelm), order='F')

  # then y
  f_p =     np.reshape(np.transpose(f_tmp2, (1,0,2,3)), (norder, ninterp**2*nelm), order='F')
  f_trans = np.reshape(np.transpose(np.reshape(trans.dot(f_p), (ninterp, ninterp, ninterp, nelm), order='F'), (1,0,2,3)), (ninterp**3, nelm),        order='F')
  toc('trans')

  #f_p = None; f_tmp2 = None; f_tmp = None; gc.collect()

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
  frac_root = np.divide(np.mod(root, shape).astype(np.float64), shape)
  ind = 0
  if frac_root[0] >= .5:
    ind = ind + 1
  if frac_root[1] >= .5:
    ind = ind + 2
  if frac_root[2] >= .5:
    ind = ind + 4
  shape = np.array(shape / 2, dtype = int)
  return ind * power + compute_index(np.mod(root,shape), shape)

def compute_Fr(h, t):
  import numpy as np
  v = [(h[i+1] - h[i-1])/(float(t[i+1])-float(t[i-1])) for i in range(1,len(h)-1)]
  v.insert(0,0.); v.append(0.)

  Fr = np.array(v) 
  return Fr

def compute_alpha(h, t):
  v = [(h[i+1] - h[i-1])/(float(t[i+1])-float(t[i-1])) for i in range(1,len(h)-1)]
  v.insert(0,0.); v.append(0.)
  alpha = [v[i]*v[i]/(4*h[i]) for i in range(len(v))]
  return alpha

def compute_alpha_quadfit(h,t):
  import numpy as np
  window = 5
  alpha = np.zeros(len(h))
  for i in range(window, len(h) - window):
    c = np.polyfit(t[i-window:i+window+1], h[i-window:i+window+1], 2)
    alpha[i] = c[0]
  return alpha

def compute_reynolds(h, t):
  v = [(h[i+1] - h[i-1])/(float(t[i+1])-float(t[i-1])) for i in range(1,len(h)-1)]
  v.insert(0,0.); v.append(0.)
  alpha = [v[i]*h[i] for i in range(len(v))]
  return alpha

def extract_dict(results):
  import numpy as np
  results_with_times = sorted([[float(elm[0]), elm[1]] for elm in results.items()])
  times, vals = zip(*results_with_times)
  times = np.array(times, dtype=np.float64)

  # Numerical stability plot
  PeCs  = np.array([d['PeCell'] for d in vals])
  TMaxs = np.array([d['TAbs']   for d in vals])
  Totals = np.array([d['Total']   for d in vals])
  hs_cabot = [d['h_cabot'] for d in vals]
  hs_visual = [d['h_visual'] for d in vals]
  hs_fit = [d['h_fit'] for d in vals]
  Xi = [d['Xi'] for d in vals]

  return times, PeCs, TMaxs, Totals, hs_cabot, hs_visual, hs_fit, Xi
