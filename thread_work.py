
def tprocess(job):
  import gc
  import numpy as np
  from my_utils import find_root, lagrange_matrix
  from my_utils import transform_field_elements
  from my_utils import transform_position_elements
  from tictoc import tic, toc

  lock = job[0]
  input_file = job[1]
  params = job[2]
  data = job[3]
  args = job[4]

  ans = {}
  ans['TMax']   = 0.
  ans['TMin']   = 0.
  ans['UAbs']   = 0.
  ans['dx_max'] = 0.
  extent = np.array(params['extent_mesh']) - np.array(params['root_mesh'])
  size = np.array(params['shape_mesh'], dtype=int)
  ninterp = int(args.ninterp*params['order'])
  cart = np.linspace(0.,extent[0],num=ninterp,endpoint=False)/size[0]
  norder = input_file.norder
  trans = None

  while True:
    tic()
    with lock:
      nelm, pos, vel, t = input_file.get_elem(args.block)
    toc('read')
    if nelm < 1:
      return ans

    if trans == None:
      gll  = pos[0:norder,0,0] - pos[0,0,0]
      trans = lagrange_matrix(gll,cart)
      if args.verbose:
        print("Interpolating\n" + str(gll) + "\nto\n" + str(cart))

    #pos_trans = transform_position_elements(pos, trans, cart)
    # pos[0,:,:] is invariant under transform, and it is all we need
    pos_trans = pos[0,:,:]
    #pos = None; gc.collect()

    # transform all the fields at once
    hunk = np.concatenate((t, vel[:,:,0], vel[:,:,1], vel[:,:,2]), axis=1)
    hunk_trans = transform_field_elements(hunk, trans, cart)
    t_trans, ux_trans, uy_trans, uz_trans = np.split(hunk_trans, 4, axis=1)
    #t, vel = None, None; gc.collect()

    # Save some results pre-renorm
    max_speed = np.sqrt(np.max(np.square(ux_trans) + np.square(uy_trans) + np.square(uz_trans)))
    ans['TMax']   = float(max(ans['TMax'], np.amax(t_trans)))
    ans['TMin']   = float(min(ans['TMin'], np.amin(t_trans)))
    ans['UAbs']   = float(max(ans['UAbs'], max_speed))
    ans['dx_max'] = float(max(ans['dx_max'], np.max(gll[1:] - gll[0:-1])))

    # Renorm t -> [0,1]
    tic()
    Tt_low = -params['atwood']/2.; Tt_high = params['atwood']/2.
    t_trans = (t_trans - Tt_low)/(Tt_high - Tt_low)
    t_trans = np.maximum(t_trans, 0.)
    t_trans = np.minimum(t_trans, 1.)
    toc('renorm')

    # stream the elements into the grid structure
    data.add(pos_trans, t_trans, ux_trans, uy_trans, uz_trans)
    #pos_trans, t_trans, ux_trans, uy_trans, uz_trans = None, None, None, None, None; gc.collect() 

