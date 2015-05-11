from utils.struct import Struct

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

  from copy import deepcopy
  base = deepcopy(ans)
  # return a cleaned up version of locals

  from interfaces.nek.files import NekFile, nek_fname
  njob_per_file = max(1+int((args.thread-1) / abs(int(params["io_files"]))),1)
  jobs = []
  for j in range(abs(int(params["io_files"]))):
      fname = nek_fname(args.name, frame, j, params["io_files"])
      ans['fname'] = fname
      if np.prod(np.array(params["shape_mesh"])) % abs(int(params["io_files"])) == 0:
        nelm = np.prod(np.array(params["shape_mesh"])) / abs(int(params["io_files"]))
      else:
        with open(fname, 'rb') as f:
          input_file = NekFile(f)
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
  from interfaces.nek.mesh import UniformMesh
  from interfaces.nek.files import NekFile
  from glopen import glopen

  ans = {}
  if scratch != None:
    ans = scratch
  # Objects are nicer to index than dicts
  a = Struct(ans)
  p = Struct(params)

  #with open(ans['fname'], 'rb') as f:
  if "input_file" not in ans:
    a.ofile = open(ans['fname'], 'rb')
    a.input_file = NekFile(a.ofile)
    #a.glopen = glopen(ans['fname'], 'rb', endpoint="maxhutch#alpha-admin/tmp/")
    #a.input_file = NekFile(a.glopen.__enter__())

  mesh = UniformMesh(a.input_file, params)
  mesh.load(pos, nelm_to_read)

  if last:
    a.input_file.close()
    a.ofile.close()
    #a.glopen.__exit__(None, None, None)

  # We need to union these sets
  a.red_uin = ['red_max', 'red_min', 'red_sum', 'slices']
  a.slices = []

  # We want slices centered here:
  intercept = (
               mesh.origin[0] + mesh.extent[0]/4.,
               mesh.origin[1] + mesh.extent[1]/4.,
               mesh.origin[2] + mesh.extent[2]/2.
               )

  # Min and max values, mostly for stability
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
  a.time   = a.input_file.time
  a.red_max.append('time')
  a.dx_max = float(np.max(mesh.gll[1:] - mesh.gll[:-1]))
  a.red_max.append('dx_max')

  # Total energy 
  u2 = np.square(mesh.fld('u'))
  a.Kinetic_x = mesh.int(u2)/2.
  a.u2_proj_z = mesh.slice(u2, intercept, (0,1), 'int')
  v2 = np.square(mesh.fld('v'))
  a.Kinetic_y = mesh.int(v2)/2. 
  a.v2_proj_z = mesh.slice(v2, intercept, (0,1), 'int')
  w2 = np.square(mesh.fld('w'))
  a.Kinetic_z = mesh.int(w2)/2.
  a.w2_proj_z = mesh.slice(w2, intercept, (0,1), 'int')
  a.slices += [ 'u2_proj_z',  'v2_proj_z',  'w2_proj_z']
  a.red_sum += ['Kinetic_x', 'Kinetic_y', 'Kinetic_z']
 
  a.Kinetic = a.Kinetic_x + a.Kinetic_y + a.Kinetic_z
  a.red_sum.append('Kinetic')

  a.Potential = p.g * mesh.int(
                   mesh.fld('t') * mesh.fld('z')
                              )
  a.red_sum.append('Potential')

  total_pressure = .5*(u2+v2+w2) + mesh.fld('p') - mesh.fld('t') * p.atwood * p.g * mesh.fld('z')
   
  # Take slices
  a.t_xy = mesh.slice(mesh.fld('t'), intercept, (2,))
  a.t_yz = mesh.slice(mesh.fld('t'), intercept, (0,))
  a.t_proj_z  = mesh.slice(mesh.fld('t'), intercept, (0,1), 'int')
  a.t_abs_proj_z = mesh.slice(np.abs(mesh.fld('t')), intercept, (0,1), 'int')
  a.t_sq_proj_z  = mesh.slice(np.square(mesh.fld('t')), intercept, (0,1), 'int')
  a.u_xy = mesh.slice(mesh.fld('u'), intercept, (2,))
  a.v_xy = mesh.slice(mesh.fld('v'), intercept, (2,))
  a.w_xy = mesh.slice(mesh.fld('w'), intercept, (2,))
  a.u_yz = mesh.slice(mesh.fld('u'), intercept, (0,))
  a.v_yz = mesh.slice(mesh.fld('v'), intercept, (0,))
  a.w_yz = mesh.slice(mesh.fld('w'), intercept, (0,))
  a.p_xy = mesh.slice(mesh.fld('p'), intercept, (2,))
  a.p_yz = mesh.slice(mesh.fld('p'), intercept, (0,))
  fz = mesh.fld('t') * p.atwood * p.g - mesh.dx('p', 2) 
  a.fz_xy = mesh.slice(fz, intercept, (2,))
  a.fz_yz = mesh.slice(fz, intercept, (0,))
  pflux = mesh.fld('t') * mesh.fld('w')
  pflux[mesh.fld('t') < 0] = 0.
  a.flux_proj_z = mesh.slice(pflux, intercept, (0,1), 'int')
  a.total_pressure_xy = mesh.slice(total_pressure, intercept, (2,))
  a.total_pressure_yz = mesh.slice(total_pressure, intercept, (0,))

  a.slices += [
               't_xy', 't_yz', 't_proj_z', 't_abs_proj_z', 't_sq_proj_z',
               'p_xy', 'p_yz', 'u_xy', 'v_xy', 'w_xy', 
               'u_yz', 'v_yz', 'w_yz', 'fz_xy', 'fz_yz',
               'flux_proj_z', 'total_pressure_xy', 'total_pressure_yz'
              ]

  a.w_max_z = mesh.slice(mesh.fld('w'), intercept, (0,1), np.maximum)
  a.w_min_z = mesh.slice(mesh.fld('w'), intercept, (0,1), np.minimum)
  a.red_max += ['w_max_z']
  a.red_min += ['w_min_z']

  dvdx = mesh.dx('v',0)
  dudy = mesh.dx('u',1)
  omegaz = dvdx - dudy 
  a.vorticity_xy = mesh.slice(omegaz,
                              intercept, (2,))
  a.vorticity_proj_z = mesh.slice(np.square(omegaz), intercept, (0,1), 'int')
  dwdy = mesh.dx('w',1)
  dvdz = mesh.dx('v',2)
  a.vorticity_yz = mesh.slice(dwdy - dvdz, intercept, (0,))
  a.slices += ['vorticity_xy', 'vorticity_yz', 'vorticity_proj_z']


  du2 = np.square(mesh.dx('u',0))
  dv2 = np.square(mesh.dx('v',1))
  dw2 = np.square(mesh.dx('w',2))
  a.du2_proj_z = mesh.slice(du2, intercept, (0,1), 'int')
  a.dv2_proj_z = mesh.slice(dv2, intercept, (0,1), 'int')
  a.dw2_proj_z = mesh.slice(dw2, intercept, (0,1), 'int')
  a.slices += ['du2_proj_z', 'dv2_proj_z', 'dw2_proj_z']

  diss = p.viscosity * (
        2. * (du2+dv2+dw2) 
      +  np.square(dvdx + dudy)  
      +  np.square(dwdy + dvdz)  
      +  np.square(mesh.dx('u',2) + mesh.dx('w',0))
                       )
  a.Dissipated = mesh.int(diss) * p.io_time
  a.red_sum.append('Dissipated')
  a.d_xy = mesh.slice(diss, intercept, (2,))
  a.d_yz = mesh.slice(diss, intercept, (0,))
  a.slices += ['d_xy', 'd_yz']

  a.red_sum += a.slices
  a.slices += ['w_max_z', 'w_min_z']
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

