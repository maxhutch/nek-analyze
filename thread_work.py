
def tprocess(job):
  import gc
  import numpy as np
  from my_utils import find_root, lagrange_matrix
  from my_utils import transform_field_elements
  from my_utils import transform_position_elements
  from nek import NekFile
  from tictoc import tic, toc

  elm_range = job[0]
  lock = job[1]
  fname = job[2]
  params = job[3]
  data = job[4]
  args = job[5]

  ans = {}
  ans['TMax']   = 0.
  ans['TMin']   = 0.
  ans['UAbs']   = 0.
  ans['dx_max'] = 0.
  extent = np.array(params['extent_mesh']) - np.array(params['root_mesh'])
  size = np.array(params['shape_mesh'], dtype=int)
  ninterp = int(args.ninterp*params['order'])
  cart = np.linspace(0.,extent[0],num=ninterp,endpoint=False)/size[0]

  input_file = NekFile(fname)
  norder = input_file.norder
  trans = None

  for pos in range(elm_range[0], elm_range[1], args.block):
    tic()
    nelm_to_read = min(args.block, elm_range[1] - pos)
    nelm, pos, vel, t = input_file.get_elem(args.block, pos)
    toc('read')

    if nelm < 1:
      return ans

    if trans == None:
      gll  = pos[0:norder,0,0] - pos[0,0,0]
      trans = lagrange_matrix(gll,cart)
      if args.verbose:
        print("Interpolating\n" + str(gll) + "\nto\n" + str(cart))
 
    # pos[0,:,:] is invariant under transform, and it is all we need
    pos_trans = np.transpose(pos[0,:,:])
 
    # transform all the fields at once
    hunk = np.concatenate((t, vel[:,:,0], vel[:,:,1], vel[:,:,2]), axis=1)
    hunk_trans = transform_field_elements(hunk, trans, cart)
    t_trans, ux_trans, uy_trans, uz_trans = np.split(hunk_trans, 4, axis=1)
 
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
    #with lock:
    data.add(pos_trans, t_trans, ux_trans, uy_trans, uz_trans)

  ans['data'] = data
  return ans
