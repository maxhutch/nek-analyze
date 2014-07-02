
def tprocess(job):
  import numpy as np
  from my_utils import lagrange_matrix
  from nek import NekFile
  from tictoc import tic, toc
  from MapReduce import Map

  elm_range = job[0]
  fname = job[1]
  params = job[2]
  ans = job[3]
  args = job[4]

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
      input_file.close()
      return ans

    if trans == None:
      gll  = pos[0:norder,0,0] - pos[0,0,0]
      trans = lagrange_matrix(gll,cart)
      if args.verbose:
        print("Interpolating\n" + str(gll) + "\nto\n" + str(cart))

    Map(pos, vel, t, trans, cart, gll, params, ans)

  input_file.close()
  return ans
