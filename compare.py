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
# Open results dictionary
results = []
params = []
for name in args.names:
  fname = '{:s}-results.dat'.format(name)
  with open(fname, 'r') as f:
    results.append(json.load(f))
  params.append(results[-1]["0.0"]["params"])
  params[-1]['g'] = 9.8

# Post-post processing
from my_utils import extract_dict
from my_utils import transpose_dict
times = []; PeCs = []; TMaxs = []; Totals = []; hs_cabots = []; hs_visuals = []; hs_fits = []; Xis = []
_new_results_ = []
for res in results:
  test = transpose_dict(res)
  #time, PeC, TMax, Total, hs_cabot, hs_visual, hs_fit, Xi = extract_dict(res)
  times.append(test["time"])
  PeCs.append(test["PeCell"])
  TMaxs.append(test["TMax"])
  Totals.append(test["Total"])
  hs_cabots.append(test["h_cabot"])
  hs_visuals.append(test["h_visual"])
  hs_fits.append(test["h_fit"])
  Xis.append(test["Xi"])
  _new_results_.append(test)

# Numerical stability plot
for j in range(len(args.names)):
  for i in range(TMaxs[j].shape[0]):
    if TMaxs[j][i] *2./params[j]['atwood'] > 1.:
      print("Simulation {:s} went unstable at t={:f}, PeCell={:f}+/-{:f}".format(args.names[j], times[j][i], (PeCs[j][i]+PeCs[j][i-1])/2, (PeCs[j][i]-PeCs[j][i-1])/2))
      break

#params = results[0]['0.0']['params']
image_y = 12
clip_y = 12/16.
image_x = len(args.names)*int(image_y * params[0]["shape_mesh"][0] / params[0]["shape_mesh"][2] / clip_y + .5)
image_x = max( image_x,  image_y * 1050/1680 )
image_x = min( image_x,  image_y * 1680/1050 )

ytics = [params[j]["root_mesh"][0],params[j]["extent_mesh"][0]]
xtics = [params[j]["root_mesh"][2]*clip_y, 0, params[j]["extent_mesh"][2]*clip_y]

Scs = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
ylabels = ["$1$", "$\sqrt{2}$", "$2$", "$2 \sqrt{2}$", "$4$", "$4 \sqrt{2}$", "$8$", "$8 \sqrt{2}$", "foo", "bar", "foobar", "barfoo", "quoi?"]
y2labels = ["2000", "1414", "1000", "707", "500", "354", "250", "177", "foo", "bar", "foobar", "barfoo", "quoi?"]
for i in range(args.frame, args.frame_end+1):
  fig = plt.figure(figsize=(image_y,image_x))
  #fig = plt.figure(figsize=(image_x,image_y))
  fig.text(0.5,0.93,'Scalar',horizontalalignment='center', verticalalignment='top', fontsize='xx-large')
  for j in range(len(args.names)):
    with open("{:s}{:05d}-raw.npz".format(args.names[j], i), 'rb') as f:
      npzfile = np.load(f)
      #ax = plt.subplot(1,len(args.names),j+1)
      ax = plt.subplot(len(args.names),1,j+1)
      dim = npzfile['yslice'].shape
      if args.names[j] == "Nu04D04":
       #ax.imshow(np.fliplr(npzfile['yslice'][:,int(dim[1]*.5*(1-clip_y)):int(dim[1]*.5*(1+clip_y))].transpose()), origin = 'lower',
       ax.imshow(np.flipud(npzfile['yslice'][:,int(dim[1]*.5*(1-clip_y)):int(dim[1]*.5*(1+clip_y))]), origin = 'lower',
         interpolation='bicubic',
         vmin = 0., vmax = 1.,
         aspect = 'auto',
         extent=[
                 params[j]["root_mesh"][2]*clip_y,params[j]["extent_mesh"][2]*clip_y,
                 params[j]["root_mesh"][0],params[j]["extent_mesh"][0]
                ])
      else:
       #ax.imshow(npzfile['yslice'][:,int(dim[1]*.5*(1-clip_y)):int(dim[1]*.5*(1+clip_y))].transpose(), origin = 'lower',
       ax.imshow(npzfile['yslice'][:,int(dim[1]*.5*(1-clip_y)):int(dim[1]*.5*(1+clip_y))], origin = 'lower',
         interpolation='bicubic',
         vmin = 0., vmax = 1.,
         aspect = 'auto',
         extent=[
                 params[j]["root_mesh"][2]*clip_y,params[j]["extent_mesh"][2]*clip_y,
                 params[j]["root_mesh"][0],params[j]["extent_mesh"][0]
                ])

      #ax.xaxis.tick_top()
      #ax.plot([params[j]["root_mesh"][0], params[j]["extent_mesh"][0]], [hs_visuals[j][i], hs_visuals[j][i]], linestyle='dashed', linewidth=1.0, color='w')
      plt.ylim([params[j]["root_mesh"][0],params[j]["extent_mesh"][0]])
      plt.xlim([params[j]["root_mesh"][2]*clip_y,params[j]["extent_mesh"][2]*clip_y])
      plt.ylabel(ylabels[j])
      if j == 0:
        #plt.xticks(xtics, fontsize='large')
        plt.yticks(ytics, fontsize='large')
      else:
        plt.yticks([])
        #plt.xticks([])
      if j == len(args.names)-1:
        #plt.yticks(ytics, fontsize='large')
        plt.xticks(xtics, fontsize='large')
      else:
        #plt.yticks([])
        plt.xticks([])
      #plt.xlabel("Sc = {:d}".format(Scs[j]), fontsize='xx-large')
      ax2 = ax.twinx()
      ax2.set_ylabel(y2labels[j])
      plt.yticks([])

  plt.savefig("compare{:05d}-yslice.png".format(i), bbox_inches="tight")
  plt.close(fig)


ylabels = ["$1$", "$\sqrt{2}$", "$2$", "$2 \sqrt{2}$", "$4$", "$4 \sqrt{2}$", "$8$", "$8 \sqrt{2}$", "foo", "bar", "foobar", "barfoo", "quoi?"]
for i in range(args.frame, min(args.frame_end+1, 82)):
  fig = plt.figure(figsize=(image_y,image_x))
  #fig = plt.figure(figsize=(image_x,image_y))
  fig.text(0.5,0.93,'Vorticity',horizontalalignment='center', verticalalignment='top', fontsize='xx-large')
  for j in range(len(args.names)):
    with open("{:s}{:05d}-raw.npz".format(args.names[j], i), 'rb') as f:
      npzfile = np.load(f)
      #ax = plt.subplot(1,len(args.names),j+1)
      ax = plt.subplot(len(args.names),1,j+1)
      dim = npzfile['yuzslice'].shape
      vorticity = (
                npzfile['yuzslice'][2:-1,1:-2]
              - npzfile['yuzslice'][0:-3,1:-2]
              - npzfile['yuxslice'][1:-2,2:-1]
              + npzfile['yuxslice'][1:-2,0:-3])/(2.*0.00390625)
      if j == 0:
        vort_max = np.max(np.max(vorticity))
        vort_min = np.min(np.min(vorticity))

      if args.names[j] == "Nu04D04":
       #ax.imshow(np.fliplr(npzfile['yslice'][:,int(dim[1]*.5*(1-clip_y)):int(dim[1]*.5*(1+clip_y))].transpose()), origin = 'lower',
       ax.imshow(np.flipud(vorticity[:,int(dim[1]*.5*(1-clip_y)):int(dim[1]*.5*(1+clip_y))]), origin = 'lower',
         interpolation='bicubic',
         vmin = vort_min, vmax = vort_max,
         aspect = 'auto',
         extent=[
                 params[j]["root_mesh"][2]*clip_y,params[j]["extent_mesh"][2]*clip_y,
                 params[j]["root_mesh"][0],params[j]["extent_mesh"][0]
                ])
      else:
       #ax.imshow(npzfile['yslice'][:,int(dim[1]*.5*(1-clip_y)):int(dim[1]*.5*(1+clip_y))].transpose(), origin = 'lower',
       ax.imshow(vorticity[:,int(dim[1]*.5*(1-clip_y)):int(dim[1]*.5*(1+clip_y))], origin = 'lower',
         interpolation='bicubic',
         vmin = vort_min, vmax = vort_max,
         aspect = 'auto',
         extent=[
                 params[j]["root_mesh"][2]*clip_y,params[j]["extent_mesh"][2]*clip_y,
                 params[j]["root_mesh"][0],params[j]["extent_mesh"][0]
                ])

      #ax.xaxis.tick_top()
      #ax.plot([params[j]["root_mesh"][0], params[j]["extent_mesh"][0]], [hs_visuals[j][i], hs_visuals[j][i]], linestyle='dashed', linewidth=1.0, color='w')
      plt.ylim([params[j]["root_mesh"][0],params[j]["extent_mesh"][0]])
      plt.xlim([params[j]["root_mesh"][2]*clip_y,params[j]["extent_mesh"][2]*clip_y])
      plt.ylabel(ylabels[j])
      if j == 0:
        #plt.xticks(xtics, fontsize='large')
        plt.yticks(ytics, fontsize='large')
      else:
        plt.yticks([])
        #plt.xticks([])
      if j == len(args.names)-1:
        #plt.yticks(ytics, fontsize='large')
        plt.xticks(xtics, fontsize='large')
      else:
        #plt.yticks([])
        plt.xticks([])
      #plt.xlabel("Sc = {:d}".format(Scs[j]), fontsize='xx-large')
      ax2 = ax.twinx()
      ax2.set_ylabel(y2labels[j])
      plt.yticks([])

  plt.savefig("compare{:05d}-yvslice.png".format(i), bbox_inches="tight")
  plt.close(fig)

from my_utils import make_movie
make_movie("compare%05d-yslice.png",  "compare-yslice.mkv")
make_movie("compare%05d-yvslice.png", "compare-yvslice.mkv")


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
from my_utils import compute_alpha, compute_alpha_quadfit, compute_reynolds, compute_Fr
alpha_cabots = []; alpha_visuals = []; alpha_quads = []
Fr_visuals = [];

from my_utils import find_root
thresh = 0.01
zs = np.linspace(params[0]['root_mesh'][2], 
                 params[0]['extent_mesh'][2], 
                 params[0]['shape_mesh'][2]*8, endpoint = False)
for i in range(len(args.names)):
  nframes = times[i].size
  for j in range(nframes):
    with open("{:s}{:05d}-raw.npz".format(args.names[i], j+1), 'rb') as f:
      npzfile = np.load(f)
      f_normed = npzfile['f_xy'] / (64 * params[0]['shape_mesh'][0] * params[0]['shape_mesh'][1])
      _new_results_[i]["h_visual"][j] = ( find_root(zs, f_normed, y0 = thresh)
                                        - find_root(zs, f_normed, y0 = 1-thresh)) / 2.

 

for i in range(len(args.names)):
  alpha_cabots.append(np.array(compute_alpha(hs_cabots[i],  times[i])) / (0.5*params[i]['atwood']*params[i]['g']))
  alpha_visuals.append(np.array(compute_alpha_quadfit(hs_visuals[i],  times[i])) / (0.5*params[i]['atwood']*params[i]['g']))
  alpha_quads.append(np.array(compute_alpha_quadfit(hs_cabots[i],  times[i])) / (0.5*params[i]['atwood']*params[i]['g']))
  Fr_visuals.append(np.array(compute_Fr(hs_visuals[i],  times[i])) / np.sqrt(0.5*params[i]['atwood']*params[i]['g']*params[i]['extent_mesh'][0]))

for res in _new_results_:
  res["alpha_cabot"]  = np.array(compute_alpha(        res["h_cabot"],  res["time"])) / (0.5*res["params"][0]['atwood']*res["params"][0]['g'])
  res["alpha_visual"] = np.array(compute_alpha_quadfit(res["h_visual"],  res["time"])) / (0.5*res["params"][0]['atwood']*res["params"][0]['g'])
  res["Fr_visual"] = np.array(compute_Fr(res["h_visual"], res["time"])) / np.sqrt(0.5*res["params"][0]['atwood']*res["params"][0]['g']*res["params"][0]['extent_mesh'][0])
  res["Fr_cabot"] = np.array(compute_Fr(res["h_cabot"], res["time"])) / np.sqrt(0.5*res["params"][0]['atwood']*res["params"][0]['g']*res["params"][0]['extent_mesh'][0])

second_font = 'x-large'

times = times / np.sqrt(0.5*params[0]['atwood'] * params[0]['g'] * 2 * np.pi / params[0]['extent_mesh'][0])


from plots import plot_spread

split = 4
gamma = 1./np.sqrt(np.pi*res["params"][0]['atwood']*res["params"][0]['g']/res["params"][0]['extent_mesh'][0])

tmax = max([np.max(a["time"]) for a in _new_results_]) 
hmax = max([np.max(a["h_visual"]) for a in _new_results_]) 
Fmax = max([np.max(a["Fr_visual"]) for a in _new_results_]) 
Xmax = max([np.max(a["Xi"]) for a in _new_results_]) 

tend = 84
#tend = 110
#labels = ["$Sc = 2, \quad \\nu = 2\sqrt{2}$", "$Sc = 1, \quad \\nu = 2$"]
labels = ["$Sc = 8, \quad \\nu = 8$", "$Sc = 1, \quad \\nu = 4\sqrt{2}$"]
plot_spread(_new_results_, "h_visual",  split, tend, "compare-h-full.png",
            xscale = gamma,
            xmax = tmax,
            yscale = res["params"][0]['extent_mesh'][0], 
            ymax = hmax,
            ylabel = "$h / \lambda$",
            extra_labels = labels)
plot_spread(_new_results_, "Fr_visual", split, tend, "compare-Fr-full.png", 
            xscale = gamma,
            xmax = tmax,
            ymax = Fmax,
            ylabel = "$\dot{h} /\sqrt{A g \lambda}$",
            extra_labels = labels)
plot_spread(_new_results_, "Xi",        split, tend, "compare-Xi-full.png",
            xscale = gamma,
            xmax = tmax,
            ymax = 1.,
            ylabel = "$\Xi$",
            extra_labels = labels)

labels = ["$Sc = 2, \quad \\nu = 2$"]
plot_spread(_new_results_[:-1], "h_visual",  split, tend, "compare-h-test.png",
            xscale = gamma,
            xmax = tmax,
            yscale = res["params"][0]['extent_mesh'][0], 
            ymax = hmax,
            ylabel = "$h / \lambda$",
            extra_labels = labels)
plot_spread(_new_results_[:-1], "Fr_visual", split, tend, "compare-Fr-test.png", 
            xscale = gamma,
            xmax = tmax,
            ymax = Fmax,
            ylabel = "$\dot{h} /\sqrt{A g \lambda}$",
            extra_labels = labels)
plot_spread(_new_results_[:-1], "Xi",        split, tend, "compare-Xi-test.png",
            xscale = gamma,
            xmax = tmax,
            ymax = 1.,
            ylabel = "$\Xi$",
            extra_labels = labels)
labels = []
plot_spread(_new_results_[:-2], "h_visual",  split, tend, "compare-h-train.png",
            xscale = gamma,
            xmax = tmax,
            yscale = res["params"][0]['extent_mesh'][0], 
            ymax = hmax,
            ylabel = "$h / \lambda$",
            extra_labels = labels)
plot_spread(_new_results_[:-2], "Fr_visual", split, tend, "compare-Fr-train.png", 
            xscale = gamma,
            xmax = tmax,
            ymax = Fmax,
            ylabel = "$\dot{h} /\sqrt{A g \lambda}$",
            extra_labels = labels)
plot_spread(_new_results_[:-2], "Xi",        split, tend, "compare-Xi-train.png",
            xscale = gamma,
            xmax = tmax,
            ymax = 1.,
            ylabel = "$\Xi$",
            extra_labels = labels)


plt.figure(figsize=(8,8))
ax1 = plt.subplot(1,1,1)
plt.xlabel('$t \\sqrt{A g k}$', fontsize = second_font)
plt.ylabel('$h / \\lambda$', fontsize = 'xx-large')
plt.ylim([0., max([np.max(a) for a in hs_cabots])/params[i]['extent_mesh'][0]]) 
for res in _new_results_:
  res["Sc"] = res["params"][0]["viscosity"]/res["params"][0]["conductivity"]
  res["Nu"] = res["params"][0]["viscosity"]/5e-5
  ax1.plot(res["time"]/gamma, 
          # np.array(res["h_cabot"])/res["params"][0]['extent_mesh'][0], 
           np.array(res["h_visual"])/res["params"][0]['extent_mesh'][0], 
           label="Nu = {:d}".format(int(res["Nu"]+.5))
          )
ax1.legend(loc=2, fontsize = second_font)
plt.savefig("compare-h.png", bbox_inches="tight")

plt.figure(figsize=(8,8))
ax1 = plt.subplot(1,1,1)
plt.xlabel('$t \\sqrt{A g k}$', fontsize = second_font)
plt.ylabel('$h / \\lambda$', fontsize = 'xx-large')
plt.ylim([0., max([np.max(a) for a in hs_cabots])/params[i]['extent_mesh'][0]]) 
for res in _new_results_:
  ax1.plot(res["time"]/gamma, 
          # np.array(res["h_cabot"])/res["params"][0]['extent_mesh'][0], 
           np.array(res["h_visual"])/res["params"][0]['extent_mesh'][0], 
           color = 'black'
          )
plt.savefig("compare-h-where.png", bbox_inches="tight")

plt.figure(figsize=(8,8))
ax1 = plt.subplot(1,1,1)
plt.xlabel('$t \\sqrt{A g k}$', fontsize = second_font)
plt.ylabel('$h / \\lambda$', fontsize = 'xx-large')
plt.ylim([0., max([np.max(a) for a in hs_cabots])/params[i]['extent_mesh'][0]]) 
for res in _new_results_[0:-2]:
  ax1.plot(res["time"]/gamma, 
          # np.array(res["h_cabot"])/res["params"][0]['extent_mesh'][0], 
           np.array(res["h_visual"])/res["params"][0]['extent_mesh'][0], 
           color = 'black'
          )
ax1.plot(_new_results_[-2]["time"]/gamma, 
        # np.array(res["h_cabot"])/res["params"][0]['extent_mesh'][0], 
         np.array(_new_results_[-2]["h_visual"])/res["params"][0]['extent_mesh'][0], 
         color = 'red', linewidth=3
        )
plt.savefig("compare-h-waldo.png", bbox_inches="tight")

plt.figure(figsize=(10,10))
ax2 = plt.subplot(1,1,1)
#ax2.yaxis.tick_right()
#ax2.yaxis.set_label_position("right")
y2labels = ["Re = 2000", "Re = 1414", "Re = 1000", "Re = 707", "Re = 500", "Re = 354", "Re = 250", "Re = 177", "foo", "bar", "foobar", "barfoo", "quoi?"]
plt.xlabel('$t \\sqrt{A g k}$', fontsize = second_font)
plt.ylabel('$\dot{h} / \sqrt{A g \\lambda}$', fontsize = 'xx-large')
plt.ylim([0., max([np.max(a) for a in Fr_visuals])]) 
for i in range(len(_new_results_)):
  res = _new_results_[i]
  #ax2.plot(res["time"], res["Fr_cabot"],
  if i == split:
    ax2.plot(res["time"][0:-3]/gamma, 
           res["Fr_visual"][0:-3],
           linewidth=3.0,
           label=y2labels[i]
          )

  else:
    ax2.plot(res["time"][0:-3]/gamma, 
           res["Fr_visual"][0:-3],
           label=y2labels[i]
          )
#ax2.plot([0,_new_results_[0]["time"][-1]/gamma], 
#         [1./np.sqrt(np.pi), 1./np.sqrt(np.pi)],
#          label='PF Model')
ax2.legend(loc=2, fontsize = second_font)
plt.savefig("compare-Fr.png", bbox_inches="tight")

plt.figure(figsize=(8,8))
ax2 = plt.subplot(1,1,1)
plt.xlabel('$t \\sqrt{A g k}$', fontsize = second_font)
plt.ylabel('$\dot{h} / \sqrt{A g \\lambda}$', fontsize = 'xx-large')
plt.ylim([0., max([np.max(a) for a in Fr_visuals])]) 
for i in range(len(_new_results_)):
  res = _new_results_[i]
  #ax2.plot(res["time"], res["Fr_cabot"],
  ax2.plot(res["time"][0:-3]/gamma,
           res["Fr_visual"][0:-3],
           color = 'black'
          )
#ax2.plot([0,_new_results_[0]["time"][-1]/gamma], 
#         [1./np.sqrt(np.pi), 1./np.sqrt(np.pi)],
#         color = 'green',
#         label='PF Model')
ax2.legend(loc=2, fontsize = second_font)
plt.savefig("compare-Fr-where.png", bbox_inches="tight")


plt.figure(figsize=(8,8))
ax2 = plt.subplot(1,1,1)
plt.xlabel('$t \\sqrt{A g k}$', fontsize = second_font)
plt.ylabel('$\dot{h} / \sqrt{A g \\lambda}$', fontsize = 'xx-large')
plt.ylim([0., max([np.max(a) for a in Fr_visuals])]) 
for i in range(len(_new_results_)-2):
  res = _new_results_[i]
  #ax2.plot(res["time"], res["Fr_cabot"],
  ax2.plot(res["time"][0:-3]/gamma, res["Fr_visual"][0:-3],
           color = 'black'
          )
ax2.plot(_new_results_[-2]["time"][0:-3]/gamma, 
         _new_results_[-2]["Fr_visual"][0:-3],
           color = 'red', linewidth=3
          )
#ax2.plot([0,_new_results_[0]["time"][-1]/gamma], 
#         [1./np.sqrt(np.pi), 1./np.sqrt(np.pi)],
#         color = 'green',
#         label='PF Model')
ax2.legend(loc=2, fontsize = second_font)
plt.savefig("compare-Fr-waldo.png", bbox_inches="tight")


plt.figure(figsize=(8,8))
ax3 = plt.subplot(1,1,1)
plt.xlabel('$t \\sqrt{A g k}$', fontsize = second_font)
plt.ylabel('$\Xi$', fontsize = 'xx-large', rotation='horizontal')
plt.ylim([0,1.])
for res in _new_results_:
  ax3.plot(res["time"]/gamma, res["Xi"],
           label="Nu = {:d}".format(int(res["Nu"]+.5))
          )
ax3.legend(loc=4, fontsize = second_font)
plt.savefig("compare-Xi.png", bbox_inches="tight")

plt.figure(figsize=(8,8))
ax3 = plt.subplot(1,1,1)
plt.xlabel('$t \\sqrt{A g k}$', fontsize = second_font)
plt.ylabel('$\Xi$', fontsize = 'xx-large', rotation='horizontal')
plt.ylim([0,1.])
for res in _new_results_:
  ax3.plot(res["time"]/gamma, res["Xi"],
           color = 'black'
          )
plt.savefig("compare-Xi-where.png", bbox_inches="tight")

plt.figure(figsize=(8,8))
ax3 = plt.subplot(1,1,1)
plt.xlabel('$t \\sqrt{A g k}$', fontsize = second_font)
plt.ylabel('$\Xi$', fontsize = 'xx-large', rotation='horizontal')
plt.ylim([0,1.])
for res in _new_results_[0:-2]:
  ax3.plot(res["time"]/gamma, res["Xi"],
           color = 'black'
          )
ax3.plot(_new_results_[-2]["time"]/gamma, 
         _new_results_[-2]["Xi"],
         color = 'red', linewidth=3
        )

plt.savefig("compare-Xi-waldo.png", bbox_inches="tight")


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
    
