def MRInit(args, params):
  """ Initialize MapReduce data """
  import numpy as np
  from Grid import Grid

  ans = {}
  ans['PeCell'] = 0.
  ans['ReCell'] = 0.
  ans['TAbs']   = 0.
  ans['TMax']   = 0.
  ans['TMin']   = 0.
  ans['UAbs']   = 0.
  ans['dx_max'] = 0.
  ans['data']   = Grid(args.ninterp * params['order'],
              params['root_mesh'],
              params['extent_mesh'],
              np.array(params['shape_mesh'], dtype=int) * int(args.ninterp * params['order']),
              boxes = args.boxes)
  return ans


def Map(pos, vel, t, trans, cart, gll, params, ans):
  """ Map operations onto chunk of elements """
  import numpy as np
  from my_utils import transform_field_elements
  from my_utils import transform_position_elements
  from tictoc import tic, toc


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
  ans['data'].add(pos_trans, t_trans, ux_trans, uy_trans, uz_trans)


def Reduce(whole, part):
  """ Reduce results into a single output object (dict) """
  whole['TMax']   = float(max(whole['TMax'],   part['TMax']))
  whole['TMin']   = float(min(whole['TMin'],   part['TMin']))
  whole['UAbs']   = float(max(whole['UAbs'],   part['UAbs']))
  whole['dx_max'] = float(max(whole['dx_max'], part['dx_max']))

  whole['data'].merge(part['data'])

  return 

