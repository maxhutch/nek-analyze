def zwgll(p):
  """
  computes the p+1 Gauss-Lobatto-Legendre nodes z on [-1,1]
  i.e. the zeros of the first derivative of the Legendre polynomial
  of degree p plus -1 and 1
  and the p+1 weights w
  """
  import numpy as np

  n = p + 1
  z = np.zeros(n)
  w = np.zeros(n)

  z[0] = -1; z[-1] = 1

  if p == 2:
    z[1] = 0;
  elif p > 1:
    M = np.zeros((p-1,p-1))
    for i in range(p-2):
      M[i,i+1] = 0.5 * np.sqrt( (i+1.) * (i+3.) / ( (i+1.5) * (i+2.5)))
      M[i+1,i] = M[i,i+1]
    d, v = np.linalg.eig(M)
    z[1:p] = np.sort(d)

  # compute the weights w
  w[0] = 2./(p*n)
  w[-1] = w[0]

  for i in range(1,p):
    x = z[i]
    z0 = 1.
    z1 = x

    for j in range(p-1):
      z2 = x * z1 * (2.*j+3.) / (j+2.) - z0 * (j+1.) / (j+2.)
      z0 = z1
      z1 = z2
    w[i] = 2/(p * n * z2 * z2)

  return z, w

def fd_weights_full(xx,x,m):
  """

     This routine evaluates the derivative based on all points
     in the stencils.  It is more memory efficient than "fd_weights"

     This set of routines comes from the appendix of 
     A Practical Guide to Pseudospectral Methods, B. Fornberg
     Cambridge Univ. Press, 1996.   (pff)

     Input parameters:
       xx -- point at wich the approximations are to be accurate
       x  -- array of x-ordinates:   x(0:n)
       m  -- highest order of derivative to be approxxmated at xi

     Output:
       c  -- set of coefficients c(0:n,0:m).
             c(j,k) is to be applied at x(j) when
             the kth derivative is approximated by a 
             stencil extending over x(0),x(1),...x(n).


     UPDATED 8/26/03 to account for matlab "+1" index shift.
     Follows p. 168--169 of Fornberg's book.
  """
  '''
  import numpy as np
  n = x.shape[0]
  c = np.ones((n,m+1));

  for i in range(n):
    for j in range(n): 
      if i == j:
        continue
      c[i,0] *= (xx - x[j]) / (x[i] - x[j]) 

  if m == 0:
    return c

  c[:,1] = 0.
  for i in range(n):
   for k in range(n):
     if k == i:
       continue
     tmp = 1./(x[i] - x[k])
     for j in range(n): 
       if k == j or j == i:
         continue
       tmp *= (xx - x[j]) / (x[i] - x[j]) 
     c[i,1] += tmp
  return c
  '''
  import numpy as np

  n1 = x.shape[0]
  n  = n1-1
  m1 = m+1

  c1       = 1.
  c4       = x[0] - xx

  c = np.zeros((n1,m1));
  c[0,0] = 1.;

  for i in range(n):
     mn = min(i+1,m)
     c2 = 1.;
     c5 = c4;
     c4 = x[i+1]-xx;
     for j in range(i+1):
        c3 = x[i+1]-x[j]
        c2 = c2*c3;
        for k in range(mn,0,-1):
           c[i+1,k] = c1*(k*c[i,k-1]-c5*c[i,k])/c2;
        c[i+1,0] = -c1*c5*c[i,0]/c2;
        for k in range(mn,0,-1):
           c[j,k] = (c4*c[j,k]-k*c[j,k-1])/c3;
        c[j,0] = c4*c[j,0]/c3;
     c1 = c2
  return c

def dhat(x):
  """ Compute the interpolatory derivative matrix D_ij associated  with nodes x_j such that
                ^
            w = D*u   
            -     -
     returns the derivative of u at the points x_i.
  """
  import numpy as np
  n1 = x.shape[0]
  w = np.zeros((n1,2))
  Dh = np.zeros((n1,n1));

  for i in range(n1):
    w = fd_weights_full(x[i],x,1)
    Dh[:,i] = w[:,1]

  Dh = np.transpose(Dh)
  return Dh

def interp_mat(x, y, order = 0):
  """ Compute the interpolatory derivative matrix P_ij associated with nodes x_j such that
                ^
            w = D*u   
            -     -
     returns the order-th derivative of u at the points y_i.
  """
  import numpy as np
  n1 = x.shape[0]
  n2 = y.shape[0]
  w = np.zeros((n1,order+1))
  Ph = np.zeros((n1,n2));

  for i in range(n2):
    w = fd_weights_full(y[i],x,order)
    Ph[:,i] = w[:,order]

  Ph = np.transpose(Ph)
  return Ph

def semhat(N):
  from sem import zwgll
  import numpy as np
  z, w = zwgll(N)

  Bh = np.diag(w)
  Dh = dhat(z)

  Ah = np.dot(np.dot(np.transpose(Dh),Bh), Dh)
  Ch = np.dot(Bh, Dh)

  return Ah, Bh, Ch, Dh, z, w

