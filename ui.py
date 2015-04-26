"""
User interfaces for the nek-analyze script

Currently, there is only a command line interface
"""


def command_line_ui():
  """
  Command line interface for nek-analyze

  Uses python's ArgumentParser to read the command line and then creates
  shortcuts for common argument combinations
  """

  # grab defaults from config files
  from os.path import exists, expanduser, join
  import json
  defaults = {"mapreduce" : "MapReduce", "post": "post"}
  if exists(join(expanduser("~"), ".nek-analyze.json")):
    with open(join(expanduser("~"), ".nek-analyze.json")) as f:
      defaults.update(json.load(f))

  # Define arguments
  from argparse import ArgumentParser
  p = ArgumentParser()
  p.add_argument("name",                 
                 help="Nek *.fld output file")
  p.add_argument("-f",  "--frame", type=int, default=1, 
                 help="[Starting] Frame number")
  p.add_argument("-e",  "--frame_end", type=int, default=-1,   
                 help="Ending frame number")
  p.add_argument("-s",  "--slice", action="store_true",
                 help="Display slice")
  p.add_argument("-c",  "--contour", action="store_true",     
                 help="Display contour")
  p.add_argument("-n",  "--ninterp", type=float, default = 1.,
                 help="Interpolating order")
  p.add_argument("-z",  "--mixing_zone", action="store_true",
                 help="Compute mixing zone width")
  p.add_argument("-m",  "--mixing_cdf", action="store_true",
                 help="Plot CDF of box temps")
  p.add_argument("-F",  "--Fourier", action="store_true",
                 help="Plot Fourier spectrum in x-y")
  p.add_argument("-b",  "--boxes", action="store_true",
                 help="Compute box covering numbers")
  p.add_argument("-nb", "--block", type=int, default=65536,
                 help="Number of elements to process at a time")
  p.add_argument("-nt", "--thread", type=int, default=1,
                 help="Number of threads to spawn")
  p.add_argument("-d",  "--display", action="store_true", default=False,  
                 help="Display plots with X")
  p.add_argument("-p",  "--parallel", action="store_true", default=False,
                 help="Use parallel map (IPython)")
  p.add_argument(       "--series", action="store_true", default=False,
                 help="Apply time-series analyses")
  p.add_argument("--mapreduce", default=defaults["mapreduce"],
                 help="Module containing Map and Reduce implementations")
  p.add_argument("--post", default=defaults["post"],
                 help="Module containing post_frame and post_series")
  p.add_argument("-v",  "--verbose", action="store_true", default=False,
                 help="Should I be really verbose, that is: wordy?")
  p.add_argument("--params", dest="param_path", default=None,
                 help="Location of param file")
  p.add_argument("--chest", dest="chest_path", default=None,
                 help="Location of chest directory")
  p.add_argument("--figs", dest="fig_path", default=None,
                 help="Location of figures")
  p.add_argument("--MR_init", default=None, 
                 help="MapReduce init function.  Loaded from --mapreduce if None.")
  p.add_argument("--reduce", default=None, 
                 help="Reduce function.  Loaded from --mapreduce if None.")
  p.add_argument("--map", default=None, 
                 help="Map function.  Loaded from --mapreduce if None.")
 
  # Load the arguments
  args = p.parse_args()
  if args.frame_end == -1:
    args.frame_end = args.frame
  args.series = (args.frame != args.frame_end) or args.series

  from os.path import dirname 
  if args.param_path is None:
    args.param_path = "{:s}.json".format(args.name) 
  if args.chest_path is None:
    args.chest_path = "{:s}-results".format(args.name) 
  if args.fig_path is None:
    args.fig_path = "{:s}".format(dirname(args.name)) 
 
  return args
