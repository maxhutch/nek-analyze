def MR_init(args, params):
  """ Initialize MapReduce data """
  import numpy as np
  from Grid import Grid

  params['extent'] = list(np.array(params['extent_mesh']) - np.array(params['root_mesh']))
  params['ninterp'] = int(args.ninterp*params['order'])
  if args.verbose:
    print("  Grid is ({:f}, {:f}, {:f}) [{:d}x{:d}x{:d}] with order {:d}".format(
            params['extent'][0], params['extent'][1], params['extent'][2], 
            params['shape_mesh'][0], params['shape_mesh'][1], params['shape_mesh'][2],
            params['order']))

  # base cases
  PeCell = 0.
  ReCell = 0.
  TAbs   = 0.
  TMax   = 0.
  TMin   = 0.
  UAbs   = 0.
  dx_max = 0.
  data   = Grid(args.ninterp * params['order'],
                params['root_mesh'],
                params['extent_mesh'],
                np.array(params['shape_mesh'], dtype=int) * int(args.ninterp * params['order']),
                boxes = args.boxes)

  # return a cleaned up version of locals
  ans = locals()
  del ans['np']
  del ans['Grid']
  del ans['args']

  return ans


def map_(pos, vel, p, t, params, scratch = None):
  """ Map operations onto chunk of elements """
  import numpy as np
  from my_utils import lagrange_matrix
  from my_utils import transform_field_elements
  from my_utils import transform_position_elements
  from tictoc import tic, toc

  ans = {}
  if scratch != None:
    ans = scratch

  # Let's compute the x, y, and z 1D bases
  cart_x = np.linspace(0., params['extent'][0], num=params['ninterp'],endpoint=False)/params['shape_mesh'][0]
  gll_x  = pos[0:params['order'],0,0] - pos[0,0,0]

  cart_y = np.linspace(0., params['extent'][1], num=params['ninterp'],endpoint=False)/params['shape_mesh'][1]
  gll_y  = pos[0:params['order']*params['order']:params['order'],0,1] - pos[0,0,1]

  cart_z = np.linspace(0., params['extent'][2], num=params['ninterp'],endpoint=False)/params['shape_mesh'][2]
  gll_z  = pos[0:params['order']**3:params['order']**2,0,2] - pos[0,0,2]

  # and then just use y
  cart = cart_y; gll = gll_y
  trans = lagrange_matrix(gll, cart)

  # pos[0,:,:] is invariant under transform, and it is all we need
  pos_trans = np.transpose(pos[0,:,:])

  # transform all the fields at once
  hunk = np.concatenate((p, t, vel[:,:,0], vel[:,:,1], vel[:,:,2]), axis=1)
  hunk_trans = transform_field_elements(hunk, trans, cart)
  p_trans, t_trans, ux_trans, uy_trans, uz_trans = np.split(hunk_trans, 5, axis=1)
  # Save some results pre-renorm
  max_speed = np.sqrt(np.max(np.square(ux_trans) + np.square(uy_trans) + np.square(uz_trans)))
  ans['TMax']   = float(np.amax(t_trans))
  ans['TMin']   = float(np.amin(t_trans))
  ans['UAbs']   = float( max_speed)
  ans['dx_max'] = float(np.max(gll[1:] - gll[0:-1]))

  # Renorm t -> [0,1]
  tic()
  Tt_low = -params['atwood']/2.; Tt_high = params['atwood']/2.
  t_trans = (t_trans - Tt_low)/(Tt_high - Tt_low)
  #t_trans = np.maximum(t_trans, -1.)
  #t_trans = np.minimum(t_trans, 2.)
  toc('renorm')

  # stream the elements into the grid structure
  ans['data'].add(pos_trans, p_trans, t_trans, ux_trans, uy_trans, uz_trans)

def reduce_(whole, part):
  """ Reduce results into a single output object (dict) """
  whole['TMax']   = float(max(whole['TMax'],   part['TMax']))
  whole['TMin']   = float(min(whole['TMin'],   part['TMin']))
  whole['UAbs']   = float(max(whole['UAbs'],   part['UAbs']))
  whole['dx_max'] = float(max(whole['dx_max'], part['dx_max']))

  if 'data' in whole:
    whole['data'].merge(part['data'])
  else:
    whole['data'] = part['data']

  return 

