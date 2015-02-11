
def get_fname(name, proc, frame, params):
  from math import log10
  from os.path import dirname, basename
  data_path = dirname(name)
  data_tag  = basename(name) 
  dir_width = int(log10(max(abs(params["io_files"])-1,1)))+1
  if params["io_files"] > 0:
    fname = "{:s}{:0{width}d}.f{:05d}".format(name, proc, frame, width=dir_width)
  else:
    fname = "{:s}/A{:0{width}d}/{:s}{:0{width}d}.f{:05d}".format(data_path, proc, data_tag, proc, frame, width=dir_width)
  return fname


def MR_init(args, params, frame):
  """ Initialize MapReduce data """
  import numpy as np

  params['extent'] = list(np.array(params['extent_mesh']) - np.array(params['root_mesh']))
  params['ninterp'] = int(args.ninterp*params['order'])
  if args.verbose:
    print("  Grid is ({:f}, {:f}, {:f}) [{:d}x{:d}x{:d}] with order {:d}".format(
            params['extent'][0], params['extent'][1], params['extent'][2], 
            params['shape_mesh'][0], params['shape_mesh'][1], params['shape_mesh'][2],
            params['order']))

  # base cases
  ans = {}
  ans['red_max'] = []
  ans['red_min'] = []
  ans['red_sum'] = []

  # return a cleaned up version of locals

  from interfaces.nek.files import NekFile
  njob_per_file = max(1+int((args.thread-1) / abs(int(params["io_files"]))),1)
  jobs = []
  for j in range(abs(int(params["io_files"]))):
      fname = get_fname(args.name, j, frame, params)
      input_file = NekFile(fname)
      elm_per_thread = int((input_file.nelm-1) / njob_per_file) + 1
      for i in range(njob_per_file):
          jobs.append([
              (i * elm_per_thread, min((i+1)*elm_per_thread, input_file.nelm)),
              fname,
              params,
              args,
              ans])  
      input_file.close()
  return jobs


def map_(input_file, pos, nelm_to_read, params, scratch = None):
  """ Map operations onto chunk of elements """
  import numpy as np
  from tictoc import tic, toc
  from interfaces.nek.mesh import UniformMesh

  ans = {}
  if scratch != None:
    ans = scratch

  mesh = UniformMesh(input_file, params)
  mesh.load(pos, nelm_to_read)

  # Save some results pre-renorm
  max_speed = np.sqrt(mesh.max(
                np.square(mesh.fld('u')) 
              + np.square(mesh.fld('v')) 
              + np.square(mesh.fld('w'))
                              ))

  ans['TMax']   = float(mesh.max(mesh.fld('t')))
  ans['red_max'].append('TMax')
  ans['TMin']   = float(mesh.min(mesh.fld('t')))
  ans['red_min'].append('TMin')
  ans['UAbs']   = float( max_speed)
  ans['red_max'].append('UAbs')
  ans["time"]   = input_file.time
  ans['red_max'].append('time')
  ans['dx_max'] = float(np.max(mesh.gll[1:] - mesh.gll[:-1]))
  ans['red_max'].append('dx_max')
  # Renorm t -> [0,1]
  tic()
  Tt_low = -params['atwood']/2.; Tt_high = params['atwood']/2.
  #t_trans = np.maximum(t_trans, -1.)
  #t_trans = np.minimum(t_trans, 2.)
  toc('renorm')

  # stream the elements into the grid structure
  return ans

def reduce_(whole, part):
  """ Reduce results into a single output object (dict) """
  import numpy as np
  
  whole['red_max'] = list(set(whole['red_max'] + part['red_max']))
  whole['red_min'] = list(set(whole['red_min'] + part['red_min']))
  whole['red_sum'] = list(set(whole['red_sum'] + part['red_sum']))

  for key in whole['red_max']:
    if key in whole and key in part:
      whole[key] = np.max(whole[key], part[key])
    elif key in part:
      whole[key] = part[key]
  for key in whole['red_min']:
    if key in whole and key in part:
      whole[key] = np.min(whole[key], part[key])
    elif key in part:
      whole[key] = part[key]
  for key in whole['red_sum']:
    if key in whole and key in part:
      whole[key] = np.sum(whole[key], part[key])
    elif key in part:
      whole[key] = part[key]

  return 

