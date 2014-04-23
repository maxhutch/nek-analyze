
class Grid:
  def __init__(self, shape):
    self.shape = shape
    return

  """ Unpack list of elements into single grid """
  def __init__(self, pos_elm, f_elm):
    import numpy as np
    from scipy.special import cbrt
    dx = pos_elm[1,0,0] - pos_elm[0,0,0]
    order = int(cbrt(pos_elm.shape[0]))
    origin = np.array([np.min(pos_elm[:,:,0]),
                       np.min(pos_elm[:,:,1]),
                       np.min(pos_elm[:,:,2])])
    corner = np.array([np.max(pos_elm[:,:,0]),
                       np.max(pos_elm[:,:,1]),
                       np.max(pos_elm[:,:,2])])
    self.shape = np.array((corner - origin)/dx + .5, dtype=int)+1
    ''' position grid '''
    self.x = np.zeros((self.shape[0], self.shape[1], self.shape[2], 3), order='F')
    for i in range(self.shape[0]):
      self.x[i,:,:,0] = dx*i + origin[0]
    for i in range(self.shape[1]):
      self.x[:,i,:,1] = dx*i + origin[1]
    for i in range(self.shape[2]):
      self.x[:,:,i,2] = dx*i + origin[2]

    ''' field grid '''
    self.f = np.zeros((self.shape[0], self.shape[1], self.shape[2]), order='F')
    for i in range(pos_elm.shape[1]):
      root = np.array((pos_elm[0,i,:] - origin)/dx + .5, dtype=int)
      self.f[root[0]:root[0]+order,
             root[1]:root[1]+order,
             root[2]:root[2]+order] = np.reshape(f_elm[:,i], (order,order,order), order='F')

def covering_number(grid, N):
  import numpy as np
  N = int(np.min(grid.shape / N))
  nshift = int(N/2)+1
  ans = grid.f.size
  for ii in range(0,N,nshift):
   for jj in range(0,N,nshift):
    for kk in range(0,N,nshift):
     counter = 0
     for i in range(ii,grid.shape[0],N): 
      for j in range(jj,grid.shape[1],N): 
       for k in range(kk,grid.shape[2],N): 
        box = grid.f[i:i+N+1,j:j+N+1,k:k+N+1]
        if np.max(box) * np.min(box) <= 0.:
          counter+=1
     ans = min(ans, counter)
  return ans

def fractal_dimension(grid, nsample = 25, base = 1.2):
  import numpy as np
  from scipy.stats import linregress
  cover_number = []
  nbox = []
  for i in range(1,nsample):
    n = int(base**i)
    if len(nbox) > 0 and n == nbox[-1]:
      continue
    nbox.append(n)
    cover_number.append(covering_number(grid, n))
  nbox = np.array(nbox)
  cover_number = np.array(cover_number)
  ans = linregress(np.log(nbox), np.log(cover_number))
  return ans[0]

def plot_slice(grid, contour = None, fname = None):
  import numpy as np 
  import matplotlib.pyplot as plt
  center = int(grid.shape[1]/2)
  plot_x = grid.x[:,center,:,0].ravel()
  plot_y = grid.x[:,center,:,2].ravel()
  plot_f = grid.f[:,center,:].ravel()

  image_x = 12
  image_y = int(image_x * grid.shape[2] / grid.shape[0] + .5)
  fig = plt.figure(figsize=(image_x,image_y))
  pts = 72*72*image_x * image_y
  ax1 = plt.subplot(1,1,1)
  plt.title('Grid Scatter plot')
  ax1.scatter(plot_x, plot_y, c=plot_f, s=pts/plot_x.size, marker="s", linewidths = 0.)
  plt.axis([np.min(plot_x), np.max(plot_x), np.min(plot_y), np.max(plot_y)])
  if contour != None:
    ax1.plot(grid.x[:,center,0,0], contour, 'k-')
  plt.xlabel('X')
  plt.ylabel('Z')
  if fname != None:
    plt.savefig(fname)

def mixing_zone(grid, thresh = .1):
  import numpy as np
  from my_utils import find_root

  L = np.max(grid.x[1,1,:,2]) - np.min(grid.x[1,1,:,2])

  f_xy = np.ones(grid.shape[2])
  for i in range(grid.shape[2]):
    f_xy[i] = np.average(grid.f[:,:,i])

  h = 0.
  for i in range(f_xy.shape[0]):
    if f_xy[i] < .5:
      h += 2*f_xy[i]
    else:
      h += 2*(1.-f_xy[i])
  hb = h * L / f_xy.shape[0]

  f_m = np.where(grid.f[:,:,:] < .5, grid.f[:,:,:]*2, 2*(1.-grid.f[:,:,:]))
  X = np.average(f_m)*grid.shape[2]/h

  return hb, X

