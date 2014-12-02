
def plot_spread(results, key, split, tend, fname, 
                xscale = 1., 
                yscale = 1., 
                xmax = -1,
                ymax = -1,
                ylabel = None, 
                extra_labels = None):
  import matplotlib.pyplot as plt
  import numpy as np
  second_font = 'x-large'

  if extra_labels != None:
    extra_plots = len(extra_labels)
    spread_end = -(1+extra_plots)
  else:
    extra_plots = 0

  plt.figure(figsize=(8,8))
  ax1 = plt.subplot(1,1,1)
  plt.xlabel('$t \\sqrt{A g k}$', fontsize = second_font)
  if ylabel != None:
    plt.ylabel(ylabel, fontsize = second_font)
  if ymax < 0:
    plt.ylim([0., max([np.max(a[key]) for a in results])/yscale])
  else:
    plt.ylim([0.,ymax/yscale])
  if xmax < 0:
    plt.xlim([0., max(results[-1]["time"][-1], results[-2]["time"][-1])/xscale])
  else:
    plt.xlim([0, xmax/xscale])

  vmin = results[0][key][0:tend]
  vmax = results[0][key][0:tend]
  for i in range(1,split+1):
    vmax = np.maximum(vmax, results[i][key][0:tend])
    vmin = np.minimum(vmin, results[i][key][0:tend])

  ax1.fill_between(
    results[0]["time"][0:tend]/xscale,
    vmin/yscale,
    vmax/yscale,
    hatch = '+',
    facecolor = 'white',
    alpha = 0.5,
    label = "Turbulent")

  vmin = results[split][key][0:tend]
  vmax = results[split][key][0:tend]
  for i in range(split+1,len(results) - extra_plots):
    vmax = np.maximum(vmax, results[i][key][0:tend])
    vmin = np.minimum(vmin, results[i][key][0:tend])

  ax1.fill_between(
    results[0]["time"][0:tend]/xscale,
    vmin/yscale,
    vmax/yscale,
    hatch = 'x',
    facecolor = 'white',
    alpha = 0.5,
    label = "Laminar")

  handles = []
  colors = ['red', 'blue', 'green']
  for i in range(extra_plots):
    foo, = ax1.plot(
      results[spread_end+i+1]["time"][0:-3]/xscale,
      results[spread_end+i+1][key][0:-3]/yscale,
      color = colors[i])
    handles.append(foo)

  p1 = plt.Rectangle((0, 0), 1, 1, fc="white", hatch='x')
  p2 = plt.Rectangle((0, 0), 1, 1, fc="white", hatch='+')
  plt.legend([p1, p2]+handles, ["Laminar", "Turbulent"] + extra_labels, loc=4, fontsize = second_font)

  #ax1.legend(loc=2, fontsize = second_font)
  plt.savefig(fname, bbox_inches="tight")
