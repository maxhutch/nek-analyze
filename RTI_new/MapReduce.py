
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

class Struct:
    """Masquerade a dictionary with a structure-like behavior."""
    """From gprof2dot.py"""

    def __init__(self, attrs = None):
        if attrs is None:
            attrs = {}
        self.__dict__['_attrs'] = attrs
    
    def __getattr__(self, name):
        try:
            return self._attrs[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self._attrs[name] = value

    def __str__(self):
        return str(self._attrs)

    def __repr__(self):
        return repr(self._attrs)


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
  ans['red_uin'] = []
  ans['red_max'] = []
  ans['red_min'] = []
  ans['red_sum'] = []

  # return a cleaned up version of locals

  from interfaces.nek.files import NekFile
  njob_per_file = max(1+int((args.thread-1) / abs(int(params["io_files"]))),1)
  jobs = []
  from copy import deepcopy
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
              deepcopy(ans)])  
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
  # Objects are nicer to index than dicts
  a = Struct(ans)
  p = Struct(params)

  mesh = UniformMesh(input_file, params)
  mesh.load(pos, nelm_to_read)

  # We need to union these sets
  a.red_uin = ['red_max', 'red_min', 'red_sum', 'slices']

  # Save some results pre-renorm
  max_speed = np.sqrt(mesh.max(
                np.square(mesh.fld('u')) 
              + np.square(mesh.fld('v')) 
              + np.square(mesh.fld('w'))
                              ))

  a.TMax   = float(mesh.max(mesh.fld('t')))
  a.red_max.append('TMax')
  a.TMin   = float(mesh.min(mesh.fld('t')))
  a.red_min.append('TMin')
  a.UAbs   = float( max_speed)
  a.red_max.append('UAbs')
  a.time   = input_file.time
  a.red_max.append('time')
  a.dx_max = float(np.max(mesh.gll[1:] - mesh.gll[:-1]))
  a.red_max.append('dx_max')

  
  a.Kinetic = mesh.int(
                 np.square(mesh.fld('u'))
               + np.square(mesh.fld('v'))
               + np.square(mesh.fld('w'))
                       )  
  a.red_sum.append('Kinetic')

  a.Potential = p.g * mesh.int(
                   mesh.fld('t') * mesh.fld('z')
                                        )
  a.red_sum.append('Potential')
   
  # Take slices
  intercept = (mesh.corner[0]/4.,mesh.corner[1]/4.,0)
  omegaz = mesh.dx('v',0) - mesh.dx('u',1)
  a.vorticity_xy = mesh.slice(omegaz,
                              intercept, (2,))
  a.vorticity_proj_z = mesh.slice(np.square(omegaz), intercept, (0,1), np.add)
  a.vorticity_yz = mesh.slice(mesh.dx('w',1) - mesh.dx('v',2),
                              intercept, (0,))
  a.t_xy = mesh.slice(mesh.fld('t'), intercept, (2,))
  a.t_yz = mesh.slice(mesh.fld('t'), intercept, (0,))
  a.t_proj_yz = mesh.slice(mesh.fld('t'), intercept, (0,1), np.add)
  a.w_xy = mesh.slice(mesh.fld('w'), intercept, (2,))
  a.w_yz = mesh.slice(mesh.fld('w'), intercept, (0,))
  a.p_xy = mesh.slice(mesh.fld('p'), intercept, (2,))
  a.p_yz = mesh.slice(mesh.fld('p'), intercept, (0,))
  a.slices = ['vorticity_xy', 'vorticity_yz', 't_xy', 't_yz', 't_proj_yz',
              'p_xy', 'p_yz', 'w_xy', 'w_yz', 'vorticity_proj_z']

  diss = p.viscosity * (
        2. * (np.square(mesh.dx('u',0)) + np.square(mesh.dx('v',1)) + np.square(mesh.dx('w',2))) 
      +  np.square(mesh.dx('v',0) + mesh.dx('u',1))  
      +  np.square(mesh.dx('w',1) + mesh.dx('v',2))  
      +  np.square(mesh.dx('u',2) + mesh.dx('w',0))
                       )
  a.Dissipated = mesh.int(diss) * p.io_time
  a.red_sum.append('Dissipated')
  a.d_xy = mesh.slice(diss, intercept, (2,))
  a.d_yz = mesh.slice(diss, intercept, (0,))
  a.slices += ['d_xy', 'd_yz']

  a.red_sum += a.slices
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
      if isinstance(whole, np.ndarray):
        whole[key] = np.max(whole[key], part[key])
      else:
        whole[key] = max(whole[key], part[key])
    elif key in part:
      whole[key] = part[key]
  for key in whole['red_min']:
    if key in whole and key in part:
      whole[key] = np.min(whole[key], part[key])
    elif key in part:
      whole[key] = part[key]
  for key in whole['red_sum']:
    if key in whole and key in part:
      whole[key] = whole[key] + part[key]
    elif key in part:
      whole[key] = part[key]

  return 

