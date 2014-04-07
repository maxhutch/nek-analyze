
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

