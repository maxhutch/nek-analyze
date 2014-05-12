#!/usr/bin/python3
import numpy as np
import json
from os.path import exists
from process_work import process
import time

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("name",                 help="Nek *.fld output file")
parser.add_argument("-f",  "--frame",       help="[Starting] Frame number", type=int, default=1)
parser.add_argument("-e",  "--frame_end",   help="Ending frame number", type=int, default=-1)
parser.add_argument("-s",  "--slice",       help="Display slice", action="store_true")
parser.add_argument("-c",  "--contour",     help="Display contour", action="store_true")
parser.add_argument("-n",  "--ninterp",     help="Interpolating order", type=float, default = 1.)
parser.add_argument("-z",  "--mixing_zone", help="Compute mixing zone width", action="store_true")
parser.add_argument("-m",  "--mixing_cdf",  help="Plot CDF of box temps", action="store_true")
parser.add_argument("-F",  "--Fourier",     help="Plot Fourier spectrum in x-y", action="store_true")
parser.add_argument("-b",  "--boxes",       help="Compute box covering numbers", action="store_true")
parser.add_argument("-nb", "--block",       help="Number of elements to process at a time", type=int, default=65536)
parser.add_argument("-nt", "--thread",       help="Number of threads to spawn", type=int, default=1)
parser.add_argument("-d",  "--display",     help="Display plots with X", action="store_true", default=False)
parser.add_argument("-v",  "--verbose",     help="Should I be really verbose, that is: wordy?", action="store_true", default=False)
args = parser.parse_args()
if args.frame_end == -1:
  args.frame_end = args.frame
args.series = (args.frame != args.frame_end)

if not args.display:
  import matplotlib
  matplotlib.use('Agg')
import matplotlib.pyplot as plt

""" Load the data """
# Load params
with open("{:s}.json".format(args.name), 'r') as f:
  params = json.load(f)
params['g'] = 9.8

# Schedule the frames
jobs = [[args, i] for i in range(args.frame, args.frame_end+1)]
start_time = time.time()
if len(jobs) > 2:
  from IPython.parallel import Client
  p = Client(profile='default')
  pmap = p.load_balanced_view().map_async
  stuff = pmap(process, jobs)
else:
  stuff =  map(process, jobs)

# Open results dictionary
fname = '{:s}-results.dat'.format(args.name)
results = {}
if exists(fname):
  with open(fname, 'r') as f:
    results = json.load(f)

# Insert new results into the dictionary
for i, res in enumerate(stuff):
  if res[0] in results:
    results[res[0]] = dict(list(results[res[0]].items()) + list(res[1].items()))
  else:
    results[res[0]] = res[1]
  with open(fname, 'w') as f:
    json.dump(results,f)
  run_time = time.time() - start_time
  print("Processed {:d}th frame after {:f}s ({:f} fps)".format(i, run_time, (i+1)/run_time)) 

# Post-post processing
if args.series: 
  results_with_times = sorted([[float(elm[0]), elm[1]] for elm in results.items()])
  times, vals = zip(*results_with_times)
  times = np.array(times, dtype=np.float64)

  # Numerical stability plot
  from my_utils import find_root
  PeCs  = np.array([d['PeCell'] for d in vals])
  TMaxs = np.array([d['TAbs']   for d in vals])
  for i in range(TMaxs.shape[0]):
    if TMaxs[i] *2./params['atwood'] > 1.:
      print("Simulation went unstable at t={:f}, PeCell={:f}+/-{:f}".format(times[i], (PeCs[i]+PeCs[i-1])/2, (PeCs[i]-PeCs[i-1])/2))
      break

  plt.figure()
  ax1 = plt.subplot(1,1,1)
  plt.xlabel('Time (s)')
  ax1.plot(times, PeCs,                      label='Cell Peclet')
  ax1.plot(times, TMaxs*2./params['atwood'], label='max(T)/max(T0)')
  plt.legend(loc=2)
  plt.savefig("{:s}-stability.png".format(args.name))

  # Make a bunch of movies
  from os import devnull
  from subprocess import call
  foo = open(devnull, 'w')
  if args.slice:
    call("rm -f "+args.name+"-slice.mkv", shell=True)
    call("avconv -f image2 -i "+args.name+"%05d-slice.png -c:v h264 "+args.name+"-slice.mkv", shell=True, stdout = foo, stderr = foo)
  if args.mixing_cdf:
    call("rm -f "+args.name+"-cdf.mkv", shell=True)
    call("avconv -f image2 -i "+args.name+"%05d-cdf.png -c:v h264 "+args.name+"-cdf.mkv", shell=True, stdout = foo, stderr = foo)
  if args.Fourier:
    call("rm -f "+args.name+"-spectrum.mkv", shell=True)
    call("avconv -f image2 -i "+args.name+"%05d-spectrum.png -c:v h264 "+args.name+"-spectrum.mkv", shell=True, stdout = foo, stderr = foo) 
  foo.close()

  # mixing zone analysis
  if args.mixing_zone: 
    from my_utils import compute_alpha
    hs_cabot = [d['h_cabot'] for d in vals]
    alpha_cabot = np.array(compute_alpha(hs_cabot, times)) / (params['atwood']*params['g'])

    hs_visual = [d['h_visual'] for d in vals]
    alpha_visual = np.array(compute_alpha(hs_visual, times)) / (params['atwood']*params['g'])

    plt.figure()
    ax1 = plt.subplot(1,3,1)
    plt.xlabel('Time (s)')
    plt.ylabel('h (m)')
    plt.ylim([0., max(hs_visual)])
    ax1.plot(times, hs_cabot, times, hs_visual)

    ax2 = plt.subplot(1,3,2)
    plt.ylim([0., max(alpha_visual)])
    ax2.plot(times, alpha_cabot, label='Cabot')
    ax2.plot(times, alpha_visual, label='Visual')
    plt.legend(loc=2)
    plt.xlabel('Time (s)')
    plt.ylabel('alpha')

    Xs = [d['Xi'] for d in vals]
    ax3 = plt.subplot(1,3,3)
    plt.ylim([0.,1.])
    ax3.plot(times, Xs)
    plt.xlabel('Time (s)')
    plt.ylabel('Xi')
    plt.savefig("{:s}-h.png".format(args.name))

    plt.figure()
    Ps = [d['P'] for d in vals]
    Ks = [d['K'] for d in vals]
    ax1 = plt.subplot(1,1,1)
    ax1.plot(times, np.divide(Ps, np.square(hs_cabot)), label='Potential')
    ax1.plot(times, np.divide(Ks, np.square(hs_cabot)), label='Kinetic')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy / h^2')
    plt.legend(loc=2)
    plt.savefig("{:s}-energy.png".format(args.name))

if  args.contour:
    cont1 = np.load("{:s}-cont{:d}.npy".format(args.name, 1))
    cont2 = np.load("{:s}-cont{:d}.npy".format(args.name, 2))
    modes = np.load("{:s}-modes.npy".format(args.name))
    kont1 = np.fft.rfft2(cont1)/cont1.size
    kont2 = np.fft.rfft2(cont2)/cont1.size
    plt.figure()
    ax1 = plt.subplot(1,1,1)
    ax1.plot(modes.ravel(), np.abs(kont1).ravel(), 'o')
    ax1.plot(modes.ravel(), np.abs(kont2).ravel(), 'o')
    plt.figure()
    ax1 = plt.subplot(1,1,1)
    ax1.plot(modes.ravel(), np.log(np.divide(np.abs(kont2), np.abs(kont1))).ravel()/0.01, 'bo') 
#    plt.xscale('log')
#    plt.yscale('log')

if args.display:
  plt.show()
    
