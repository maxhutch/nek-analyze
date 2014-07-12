
def tprocess(job):
  import numpy as np
  from nek import NekFile
  from tictoc import tic, toc
  from MapReduce import Map, Reduce

  elm_range, fname, params, ans, args = job

  from copy import deepcopy
  res = deepcopy(ans)

  input_file = NekFile(fname)

  for pos in range(elm_range[0], elm_range[1], args.block):
    tic()
    nelm_to_read = min(args.block, elm_range[1] - pos)
    nelm, x, u, p, t = input_file.get_elem(nelm_to_read, pos)
    toc('read')

    if nelm < 1:
      input_file.close()
      return res

    Map(x, u, p, t, params, ans)
    Reduce(res, ans)

  input_file.close()
  return res
