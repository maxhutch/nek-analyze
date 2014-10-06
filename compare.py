#!/usr/bin/env python3
import numpy as np
import json
from os.path import exists
import time

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("names", nargs='+',     help="Nek *.fld output file")
parser.add_argument("-f",  "--frame",       help="[Starting] Frame number", type=int, default=1)
parser.add_argument("-e",  "--frame_end",   help="Ending frame number", type=int, default=-1)
args = parser.parse_args()
if args.frame_end == -1:
  args.frame_end = args.frame
args.display = False 

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

#params = results[0]['0.0']['params']
image_y = 8
image_x = 5*int(image_y * params[0]["shape_mesh"][0] / params[0]["shape_mesh"][2] + .5)
image_x = max( image_x,  image_y * 1050/1680 )
image_x = min( image_x,  image_y * 1680/1050 )

xtics = [params[j]["root_mesh"][0],params[j]["extent_mesh"][0]]
ytics = [params[j]["root_mesh"][2], 0, params[j]["extent_mesh"][2]]

Scs = [1, 7, 1]

for i in range(args.frame, args.frame_end+1):
  fig = plt.figure(figsize=(image_x,image_y))
  fig.text(0.5,0.93,'Scalar',horizontalalignment='center', verticalalignment='top', fontsize='xx-large')
  for j in range(len(args.names)):
    with open("{:s}{:05d}-raw.npz".format(args.names[j], i), 'rb') as f:
      npzfile = np.load(f)
      ax = plt.subplot(1,len(args.names),j+1)
      ax.imshow(npzfile['yslice'].transpose(), origin = 'lower',
        interpolation='bicubic',
        vmin = 0., vmax = 1.,
        aspect = 'auto',
        extent=[params[j]["root_mesh"][0],params[j]["extent_mesh"][0],
                params[j]["root_mesh"][2],params[j]["extent_mesh"][2]]
               )
      #ax.plot([params[j]["root_mesh"][0], params[j]["extent_mesh"][0]], [hs_visuals[j][i], hs_visuals[j][i]], linestyle='dashed', linewidth=1.0, color='w')
      plt.xlim([params[j]["root_mesh"][0],params[j]["extent_mesh"][0]])
      plt.ylim([params[j]["root_mesh"][2],params[j]["extent_mesh"][2]])
      if j == 0:
        plt.xticks(xtics, fontsize='large')
      else:
        plt.xticks([])
      if j == 0:
        plt.yticks(ytics, fontsize='large')
      else:
        plt.yticks([])
      plt.xlabel("Sc = {:d}".format(Scs[j]), fontsize='xx-large')

  plt.savefig("compare{:05d}-yslice.png".format(i), bbox_inches="tight")
  plt.close(fig)

for i in range(args.frame, args.frame_end+1):
  fig = plt.figure(figsize=(image_x,image_y))
  fig.text(0.5,0.93,'Vertical velocity',horizontalalignment='center', verticalalignment='top', fontsize='xx-large')
  for j in range(len(args.names)):
    with open("{:s}{:05d}-raw.npz".format(args.names[j], i), 'rb') as f:
      npzfile = np.load(f)
      ax = plt.subplot(1,len(args.names),j+1)
      ax.imshow(npzfile['yuzslice'].transpose(), origin = 'lower',
        interpolation='bicubic',
        aspect = 'auto',
        extent=[params[j]["root_mesh"][0],params[j]["extent_mesh"][0],
                params[j]["root_mesh"][2],params[j]["extent_mesh"][2]]
               )
      if j == 0:
        plt.xticks(xtics, fontsize='large')
      else:
        plt.xticks([])
      if j == 0:
        plt.yticks(ytics, fontsize='large')
      else:
        plt.yticks([])
      plt.xlabel("Sc = {:d}".format(Scs[j]), fontsize='xx-large')


  plt.savefig("compare{:05d}-yuzslice.png".format(i), bbox_inches="tight")
  plt.close(fig)

for i in range(args.frame, args.frame_end+1):
  fig = plt.figure(figsize=(image_x,image_y))
  fig.text(0.5,0.93,'Vorticity',horizontalalignment='center', verticalalignment='top', fontsize='xx-large')
  for j in range(len(args.names)):
    with open("{:s}{:05d}-raw.npz".format(args.names[j], i), 'rb') as f:
      npzfile = np.load(f)
      ax = plt.subplot(1,len(args.names),j+1)
      ax.imshow(npzfile['yvslice'].transpose(), origin = 'lower',
        interpolation='bicubic',
        aspect = 'auto',
        extent=[params[j]["root_mesh"][0],params[j]["extent_mesh"][0],
                params[j]["root_mesh"][2],params[j]["extent_mesh"][2]]
               )
      if j == 0:
        plt.xticks(xtics, fontsize='large')
      else:
        plt.xticks([])
      if j == 0:
        plt.yticks(ytics, fontsize='large')
      else:
        plt.yticks([])
      plt.xlabel("Sc = {:d}".format(Scs[j]), fontsize='xx-large')


  plt.savefig("compare{:05d}-yvslice.png".format(i), bbox_inches="tight")
  plt.close(fig)



from os import devnull
from subprocess import call
foo = open(devnull, 'w')
codec = "mpeg4"
options = "-mbd rd -flags +mv4+aic -trellis 2 -cmp 2 -subcmp 2 -g 300 -qscale 5 -pass 1 -strict experimental"
call("rm -f compare-yslice.mkv", shell=True)
call("avconv -f image2 -i compare%05d-yslice.png -c:v {:s} {:s} compare-yslice.mkv".format(codec, options), shell=True, stdout = foo, stderr = foo)
call("rm -f compare-yuzslice.mkv", shell=True)
call("avconv -f image2 -i compare%05d-yuzslice.png -c:v {:s} {:s} compare-yuzslice.mkv".format(codec, options), shell=True, stdout = foo, stderr = foo)
call("rm -f compare-yvslice.mkv", shell=True)
call("avconv -f image2 -i compare%05d-yvslice.png -c:v {:s} {:s} compare-yvslice.mkv".format(codec, options), shell=True, stdout = foo, stderr = foo)



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

colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yelow']

# mixing zone analysis
if True: 
  from my_utils import compute_alpha, compute_alpha_quadfit, compute_reynolds, compute_Fr
  alpha_cabots = []; alpha_visuals = []; alpha_quads = []
  Fr_visuals = [];

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
    Fr_visuals.append(np.array(compute_Fr(hs_visuals[i],  times[i])) / np.sqrt(params[i]['atwood']*params[i]['g']*params[i]['extent_mesh'][0]))

  second_font = 'x-large'

  times = times / np.sqrt(params[0]['atwood'] * params[0]['g'] * 2 * np.pi / params[0]['extent_mesh'][0])

  plt.figure()
  ax1 = plt.subplot(1,2,1)
  plt.xlabel('$t / \\sqrt{A g k}$', fontsize = second_font)
  plt.ylabel('$h / \\lambda$', fontsize = 'xx-large')
  plt.ylim([0., max([np.max(a) for a in hs_cabots])/params[i]['extent_mesh'][0]]) 
  for i in range(len(args.names)):
    ax1.plot(times[i], np.array(hs_visuals[i])/params[i]['extent_mesh'][0], label="Sc = {:d}".format(Scs[i]))
  ax1.legend(loc=2, fontsize = second_font)

  ax2 = plt.subplot(1,2,2)
  ax2.yaxis.tick_right()
  ax2.yaxis.set_label_position("right")
  plt.xlabel('$t / \\sqrt{A g k}$', fontsize = second_font)
  plt.ylabel('$\dot{h} / \sqrt{A g \\lambda}$', fontsize = 'xx-large')
  plt.ylim([0., max([np.max(a) for a in Fr_visuals])]) 
  for i in range(len(args.names)):
    ax2.plot(times[i], Fr_visuals[i])
  plt.savefig("compare-h.png", bbox_inches="tight")

  plt.figure()
  ax3 = plt.subplot(1,1,1)
  plt.xlabel('$t / \\sqrt{A g k}$', fontsize = second_font)
  plt.ylabel('$\Xi$', fontsize = 'xx-large', rotation='horizontal')
  plt.ylim([0,1.])
  for i in range(len(args.names)):
    ax3.plot(times[i], Xis[i], label="Sc = {:d}".format(Scs[i]))
  ax3.legend(loc=2, fontsize = second_font)
  plt.savefig("compare-Xi.png", bbox_inches="tight")

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

if False:
  plt.show()
    
