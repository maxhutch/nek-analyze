from utils.struct import Struct

def MR_init(args, params, frame):
  """ Initialize MapReduce data """
  import numpy as np

  # base cases
  ans = {}
  ans['red_uin'] = []
  ans['red_max'] = []
  ans['red_min'] = []
  ans['red_sum'] = []

  from copy import deepcopy
  base = deepcopy(ans)
  # return a cleaned up version of locals

  from interfaces.nek.files import NekFld, nek_fname
  njob_per_file = max(1+int((args.thread-1) / abs(int(params["io_files"]))),1)
  jobs = []
  for j in range(abs(int(params["io_files"]))):
      ans['fname'] = []
      for i in range(params["snapshots"]):
        name_i = "{:s}-{:d}".format(args.name, i)
        fname = nek_fname(name_i, frame, j, params["io_files"])
        ans['fname'].append(fname)
      if np.prod(np.array(params["shape_mesh"])) % abs(int(params["io_files"])) == 0:
        nelm = np.prod(np.array(params["shape_mesh"])) / abs(int(params["io_files"]))
      else:
        with open(fname, 'rb') as f:
          input_file = NekFld(f)
          nelm = input_file.nelm
          input_file.close()
      
      elm_per_thread = int((nelm-1) / njob_per_file) + 1
      for i in range(njob_per_file):
          jobs.append([
              (i * elm_per_thread, min((i+1)*elm_per_thread, nelm)),
              params,
              args,
              deepcopy(ans)])  
  return jobs, base


def map_(pos, nelm_to_read, params, scratch = None, last = False):
  """ Map operations onto chunk of elements """
  import numpy as np
  from tictoc import tic, toc
  from interfaces.nek.mesh import GeneralMesh
  from interfaces.nek.files import NekFld
  from glopen import glopen

  ans = {}
  if scratch != None:
    ans = scratch
  # Objects are nicer to index than dicts
  a = Struct(ans)
  p = Struct(params)

  from tictoc import tic, toc
  tic()
  #with open(ans['fname'], 'rb') as f:
  if "input_file" not in ans:
      a.ofile = []
      a.input_file = []
      for fname in ans['fname']:
        a.ofile.append(open(fname, 'rb'))
        a.input_file.append(NekFld(a.ofile[-1]))
    #a.glopen = glopen(ans['fname'], 'rb', endpoint="maxhutch#alpha-admin/tmp/")
    #a.input_file = NekFile(a.glopen.__enter__())

  meshs = []
  for input_file in a.input_file:
    meshs.append(GeneralMesh(input_file, params))
    meshs[-1].load(pos, nelm_to_read)
  
  if last:
    for input_file, ofile in zip(a.input_file, a.ofile):
      input_file.close()
      ofile.close()
    #a.glopen.__exit__(None, None, None)
  toc('load')
  tic()
  # We need to union these sets
  a.red_uin = ['red_max', 'red_min', 'red_sum', 'slices']
  a.slices = []

  a.time   = a.input_file[0].time
  a.red_max.append('time')

  a.overlap = np.zeros((p.snapshots, p.snapshots))
  for i in range(p.snapshots):
    #vel_mag_i = np.sqrt(np.square(meshs[i].fld('u')) + np.square(meshs[i].fld('v')) + np.square(meshs[i].fld('w')))
    for j in range(p.snapshots):
      foo = meshs[i].fld('u') * meshs[j].fld('u') + meshs[i].fld('v') * meshs[j].fld('v') + meshs[i].fld('w') * meshs[j].fld('w')
      #vel_mag_j = np.sqrt(np.square(meshs[j].fld('u')) + np.square(meshs[j].fld('v')) + np.square(meshs[j].fld('w')))
      #a.overlap[i,j] = meshs[0].int(vel_mag_i * vel_mag_j)
      a.overlap[i,j] = meshs[0].int(foo)

  ones = np.ones(meshs[0].fld('x').shape)
  a.volume = meshs[0].int(ones)
  a.x_min = meshs[0].min(meshs[0].fld('x'))
  a.y_min = meshs[0].min(meshs[0].fld('y'))
  a.z_min = meshs[0].min(meshs[0].fld('z'))
  a.x_max = meshs[0].max(meshs[0].fld('x'))
  a.y_max = meshs[0].max(meshs[0].fld('y'))
  a.z_max = meshs[0].max(meshs[0].fld('z'))

  a.red_sum.append("overlap")
  a.red_sum.append("volume")
  a.red_min += ["x_min", "y_min", "z_min",]
  a.red_max += ["x_max", "y_max", "z_max",]
  toc('map')

  return ans

def reduce_(whole, part):
  """ Reduce results into a single output object (dict) """
  import numpy as np
  
  # Hardcode unioning the lists of things to union
  whole['red_uin'] = list(set(whole['red_uin'] + part['red_uin']))

  # Union the list of things to union
  for key in whole['red_uin']:
    if key in whole and key in part:
      whole[key] = list(set(whole[key] + part[key]))
    elif key in part:
      whole[key] = part[key]

  # Handle other types of reductions
  for key in whole['red_max']:
    if key in whole and key in part:
      if isinstance(whole[key], np.ndarray):
        whole[key] = np.maximum(whole[key], part[key])
      else:
        whole[key] = max(whole[key], part[key])
    elif key in part:
      whole[key] = part[key]
  for key in whole['red_min']:
    if key in whole and key in part:
      if isinstance(whole[key], np.ndarray):
        whole[key] = np.minimum(whole[key], part[key])
      else:
        whole[key] = min(whole[key], part[key])
    elif key in part:
      whole[key] = part[key]
  for key in whole['red_sum']:
    if key in whole and key in part:
      whole[key] = whole[key] + part[key]
    elif key in part:
      whole[key] = part[key]

  return 

