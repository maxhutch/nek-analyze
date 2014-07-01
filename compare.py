#!/usr/bin/python3
import numpy as np
import json
from os.path import exists
from process_work import process
import time

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("names", nargs='+',     help="Nek *.fld output file")
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
params = []
for name in args.names:
  with open("{:s}.json".format(name), 'r') as f:
    params.append(json.load(f))
    params[-1]['g'] = 9.8

# Open results dictionary
results = []
for name in args.names:
  fname = '{:s}-results.dat'.format(name)
  with open(fname, 'r') as f:
    results.append(json.load(f))

# Post-post processing
from my_utils import extract_dict
times = []; PeCs = []; TMaxs = []; Totals = []; hs_cabots = []; hs_visuals = []; hs_fits = []; Xis = []
for res in results:
  time, PeC, TMax, Total, hs_cabot, hs_visual, hs_fit, Xi = extract_dict(res)
  times.append(time)
  PeCs.append(PeC)
  TMaxs.append(TMax)
  Totals.append(Total)
  hs_cabots.append(hs_cabot)
  hs_visuals.append(hs_visual)
  hs_fits.append(hs_fit)
  Xis.append(Xi)

# Numerical stability plot
for j in range(len(args.names)):
  for i in range(TMaxs[j].shape[0]):
    if TMaxs[j][i] *2./params[j]['atwood'] > 1.:
      print("Simulation {:s} went unstable at t={:f}, PeCell={:f}+/-{:f}".format(args.names[j], times[j][i], (PeCs[j][i]+PeCs[j][i-1])/2, (PeCs[j][i]-PeCs[j][i-1])/2))
      break

'''
plt.figure()
ax1 = plt.subplot(1,1,1)
plt.xlabel('Time (s)')
ax1.plot(times, PeCs,                      label='Cell Peclet')
ax1.plot(times, TMaxs*2./params['atwood'], label='max(T)/max(T0)')
ax1.plot(times, Totals/np.max(np.abs(Totals)), label='avg(T)/max(avg(T))')
plt.legend(loc=2)
plt.savefig("{:s}-stability.png".format(args.name))
'''

'''
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
'''

colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yelow']

# mixing zone analysis
if args.mixing_zone: 
  from my_utils import compute_alpha, compute_alpha_quadfit, compute_reynolds
  alpha_cabots = []; alpha_visuals = []; alpha_quads = []

  '''
  args.names.append("Fit")
  times.append(times[0])
  params.append(params[0])
  gt = 0.788314 * times[0]
  hs_cabots.append(2*np.cosh(gt) * 0.001 / 64.)
  hs_visuals.append(2*np.cosh(gt) * 0.001 / 64.)
  hs_fits.append(2*np.cosh(gt) * 0.001 / 64.)
  '''

  for i in range(len(args.names)):
    alpha_cabots.append(np.array(compute_alpha(hs_cabots[i],  times[i])) / (params[i]['atwood']*params[i]['g']))
    alpha_visuals.append(np.array(compute_alpha_quadfit(hs_visuals[i],  times[i])) / (params[i]['atwood']*params[i]['g']))
    alpha_quads.append(np.array(compute_alpha_quadfit(hs_cabots[i],  times[i])) / (params[i]['atwood']*params[i]['g']))

  plt.figure()
  ax1 = plt.subplot(1,1,1)
  plt.xlabel('Time (s)')
  plt.ylabel('\\alpha')
  plt.ylim([0., max([np.max(a) for a in alpha_quads + alpha_cabots])]) 
  for i in range(len(args.names)):
    ax1.plot(times[i], alpha_cabots[i], color=colors[i], label=args.names[i])
    ax1.plot(times[i], alpha_quads[i], color=colors[i], linestyle='dashed')
    ax1.plot(times[i], alpha_visuals[i], color=colors[i], linestyle='dotted')
  plt.legend(loc=1)

  plt.figure()
  ax1 = plt.subplot(1,1,1)
  plt.xlabel('Time (s)')
  plt.ylabel('h (m)')
  plt.ylim([0., max([np.max(a) for a in hs_cabots])]) 
  for i in range(len(args.names)):
    ax1.plot(times[i], hs_cabots[i])
    np.savetxt(args.names[i]+"-hc.csv", np.vstack((times[i], hs_cabots[i])).transpose(), delimiter=",")
    np.savetxt(args.names[i]+"-hv.csv", np.vstack((times[i], hs_visuals[i])).transpose(), delimiter=",")
    print(i)
    print(args.names)
    print(hs_fits[i])
    np.savetxt(args.names[i]+"-hf.csv", np.vstack((times[i], hs_fits[i])).transpose(), delimiter=",")

'''
  plt.figure()
  ax3 = plt.subplot(1,1,1)
  plt.ylim([0.,1.])
  for i in range(len(args.names)):
    ax3.plot(times[i], Xis[i])
  plt.xlabel('Time (s)')
  plt.ylabel('Xi')
'''

'''
  gamma = 0.8822
  gt = gamma * times[0]
  ax1.plot(times[0], params[0]['amp0']*gamma*gamma/params[0]['kmin'] * np.divide(np.sinh(gt) * np.sinh(gt), 4 * params[0]['atwood']*params[0]['g']*np.cosh(gt))) 
  gamma = 0.8802
  gt = gamma * times[0]
  ax1.plot(times[0], params[0]['amp0']*gamma*gamma/params[0]['kmin'] * np.divide(np.sinh(gt) * np.sinh(gt), 4 * params[0]['atwood']*params[0]['g']*np.cosh(gt))) 
'''

'''
  ax2 = plt.subplot(1,4,2)
  plt.ylim([0., max(np.max(alpha_visual),np.max(alpha_cabot))])
  ax2.plot(times, alpha_cabot, label='Cabot')
  ax2.plot(times, alpha_visual, label='Visual')
  plt.legend(loc=2)
  plt.xlabel('Time (s)')
  plt.ylabel('alpha')

  Re_visual = np.array(compute_reynolds(hs_visual, times)) / (params['viscosity'])
  ax4 = plt.subplot(1,4,4)
  ax4.plot(times, Re_visual)
  plt.xlabel('Time (s)')
  plt.ylabel('Re')
'''

#  plt.savefig("{:s}-compare-h.png".format(args.name))

'''
  plt.figure()
  Ps = np.array([d['P'] for d in vals])
  Ks = np.array([d['K'] for d in vals])
  ax1 = plt.subplot(1,1,1)
  budget = (params['atwood'] * params['g'] * np.square(hs_cabot) * 
           (params["extent_mesh"][0] - params["root_mesh"][0]) *
           (params["extent_mesh"][1] - params["root_mesh"][1]) / 2.)
  ax1.plot(times, np.divide(Ps-Ps[0], budget), label='Potential')
  ax1.plot(times, np.divide(Ks      , budget), label='Kinetic')
  plt.xlabel('Time (s)')
  plt.ylabel('Energy / h^2')
  plt.legend(loc=2)
  plt.savefig("{:s}-energy.png".format(args.name))

  args.contour:
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
'''

if args.display:
  plt.show()
    
