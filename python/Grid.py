
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

def plot_slice(grid, contour = None, fname = None):
  import numpy as np 
  import matplotlib.pyplot as plt
  center = int(grid.shape[1]/2)
  plot_x = grid.x[:,center,:,0].ravel()
  plot_y = grid.x[:,center,:,2].ravel()
  plot_f = grid.f[:,center,:].ravel()


  fig = plt.figure(figsize=(12,12))
  pts = 746496.
  ax1 = plt.subplot(1,1,1)
  plt.title('Grid Scatter plot')
  ax1.scatter(plot_x, plot_y, c=plot_f, s=pts/plot_x.size, linewidths = 0.)
  plt.axis([np.min(plot_x), np.max(plot_x), np.min(plot_y), np.max(plot_y)])
  if contour != None:
    ax1.plot(grid.x[:,center,0,0], contour, 'k-')
  plt.xlabel('X')
  plt.ylabel('Z')
  if fname != None:
    plt.savefig(fname)

def mixing_zone(grid, thresh = .1, plot = False, fname = None, time = -1.):
  import numpy as np
  from my_utils import find_root
  f_xy = np.ones(grid.shape[2])
  for i in range(grid.shape[2]):
    f_xy[i] = np.average(grid.f[:,:,i])

  visual_boundary = False
  if visual_boundary:
    boundaries = (find_root(grid.x[1,1,:,2], f_xy, y0 = thresh), find_root(grid.x[1,1,:,2], f_xy, y0 = 1-thresh))
  else:
    h = 0.
    for i in range(f_xy.shape[0]):
      if f_xy[i] < .5:
        h += 2*f_xy[i]
      else:
        h += 2*(1.-f_xy[i])
    h = h * (np.max(grid.x[1,1,:,2]) - np.min(grid.x[1,1,:,2])) / f_xy.shape[0]
    boundaries = ((np.max(grid.x[1,1,:,2]) + np.min(grid.x[1,1,:,2]) + h)/2.,
                  (np.max(grid.x[1,1,:,2]) + np.min(grid.x[1,1,:,2]) - h)/2.)
  if fname != None:
    with open(fname, "a") as f:
      f.write("{:f} {:13.10f} {:13.10f}\n".format(time, boundaries[0], boundaries[1]))

  if plot:
    import matplotlib.pyplot as plt
    plt.figure()
    ax1 = plt.subplot(1,1,1)
    ax1.plot(grid.x[1,1,:,2], f_xy, 'k-')
    ax1.vlines(boundaries, (0, 1), (thresh, 1-thresh))

  return boundaries
