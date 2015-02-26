"""
Post-processing module: to be completed by user
"""

def post_series(results, params, args):
  """Post-process time-series results, outputting to screen or files.

  Keyword arguments:
  results -- dictionary of ouputs of process_work keyed by time
  params  -- dictionary of problem parameters read from {name}.json
  args    -- namespace of commandline arguments from ArgumentParser
  """

  import numpy as np
  import matplotlib
  if not args.display:
    matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  # Extract the times
  times = np.array(results[:,'frame'].keys())

  # Numerical stability plot
  PeCs  = np.array(results[:,'PeCell'].values())
  TMaxs = np.array(results[:,'TAbs'].values())
  for i in range(1,TMaxs.shape[0]):
    if TMaxs[i] > TMaxs[0]:
      print("Simulation went unstable at t={:f}, PeCell={:f}+/-{:f}".format(times[i], (PeCs[i]+PeCs[i-1])/2, (PeCs[i]-PeCs[i-1])/2))
      break

  plt.figure()
  ax1 = plt.subplot(1,1,1)
  plt.xlabel('Time (s)')
  ax1.plot(times, np.log(PeCs),          label='Log[Cell Peclet]')
  ax1.plot(times, TMaxs*2./params['atwood'], 'gx', label='max(T)/max(T0)')
  plt.ylim(ymin = 0)
  plt.legend(loc=2)
  plt.savefig("{:s}-stability.png".format(args.name))

  # Energy budget
  KEs = np.array(results[:,'Kinetic'].values())
  #KXs = np.array(results[:,'Kinetic_x'].values()) + np.array(results[:,'Kinetic_y'].values())
  #KZs = np.array(results[:,'Kinetic_z'].values())
  PEs = np.array(results[:,'Potential'].values())
  DIs = np.array(results[:,'Dissipated'].values())
  for i in range(DIs.shape[0]-1, -1, -1):
      DIs[i] = np.trapz(DIs[:i+1])
  PEs = PEs - PEs[0]
  #DFs = params["g"] * params["atwood"] * params["conductivity"] * times
  DFs = PEs - KEs - DIs
  plt.figure()
  ax1 = plt.subplot(1,1,1)
  plt.xlabel('Time (s)')
  ax1.plot(times, KEs /PEs,     label="Kinetic")
  #ax1.plot(times, KXs /PEs,     label="Kinetic_x", 'b--')
  #ax1.plot(times, KZs /PEs,     label="Kinetic_y", 'b-.')
  ax1.plot(times, DIs /PEs, label="Dissipated")
  ax1.plot(times, DFs /PEs , label="Diffused")
  plt.ylim(ymin = 0, ymax = 1)
  plt.legend(loc=2, ncol=3)
  plt.savefig("{:s}-energy.png".format(args.name))

  # Froude Number



  # Finally, stitch together frames into movies
  from utils.my_utils import make_movie
  for name in results[times[0], 'slices']:
    make_movie("{:s}-{:s}-%04d.png".format(args.name, name),  "{:s}-{:s}.mkv".format(args.name, name))

  return

import os.path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from os.path import basename
def plot_slice(data, name):

  min_size = 6
  if len(data.shape) == 2:
    fsize = np.array(data.shape) * min_size / min(data.shape)
    plt.figure(figsize=tuple(fsize.tolist()))
    ax = plt.subplot(111)
    ax.imshow(data.transpose(), origin='lower')
    plt.title(basename(name))
    plt.savefig('{:s}.png'.format(name))
    plt.close()
  elif len(data.shape) == 1:
    plt.figure(figsize=(min_size, min_size))
    ax = plt.subplot(111)
    ax.plot(data)
    plt.title(basename(name))
    plt.savefig('{:s}.png'.format(name))
    plt.close()
  return

def post_frame(ans, params, args):
  # Analysis! 
  ans['TAbs'] = max(ans['TMax'], -ans['TMin'])
  ans['PeCell'] = ans['UAbs']*ans['dx_max']/params['conductivity']
  ans['ReCell'] = ans['UAbs']*ans['dx_max']/params['viscosity']

  # Mixing height
  L = params["extent_mesh"][2] - params["root_mesh"][2]
  h = 0.
  tmax = np.max(ans["t_proj_z"])
  tmin = np.min(ans["t_proj_z"])
  tzero = (tmax + tmin) / 2
  h_cabot = 0.
  for i in range(ans["t_proj_z"].shape[0]):
    if ans["t_proj_z"][i] < tzero:
      h_cabot += (ans["t_proj_z"][i] - tmin) 
    else:
      h_cabot += (tmax - ans["t_proj_z"][i]) 
  ans["h"] = h_cabot

  zs = np.linspace(params["root_mesh"][2], params["extent_mesh"][2], ans["t_proj_z"].shape[0], endpoint = True)
  from utils.my_utils import find_root
  h_visual = ( find_root(zs, ans["t_proj_z"], y0 = tmax - (tmax - tmin)*.01)
             - find_root(zs, ans["t_proj_z"], y0 = tmin + (tmax - tmin)*0.1)) / 2.

  ans["H"] = h_visual

  return 

def plot_frame(ans, params, args):
  # Analysis! 
  if args.verbose:
    print("  Extremal temperatures {:f}, {:f}".format(ans['TMax'], ans['TMin']))
    print("  Max speed: {:f}".format(ans['UAbs']))
    print("  Cell Pe: {:f}, Cell Re: {:f}".format(ans['PeCell'], ans['ReCell']))

  for name in ans['slices']:
    plot_slice(ans[name], "{:s}-{:s}-{:04d}".format(args.name,name, ans['frame']))

  return 

