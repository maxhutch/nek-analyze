# Analysis framework Nek5000 

nek-analyze is a framework for writing analysis and visualization tools for
 Nek5000 files.  nek-analyze employes a two-tiered MapReduce model, giving it
some useful properties:
 * O(1) memory usage, tunable at run-time
 * Efficient on and off-node parallelization
 * Scipy ecosystem tools

Many common analyses can be easily expressed as MapReduce with a bit of 
post-processing:
 * Single variable integrals (e.g. int dV)
 * Global masked min/max
 * Dense analysis on lines and slices (e.g. 2D FFT, plotting)
 * Visualization of slices
 * Marking iso-surfaces

However, some analyses fall outside this model:
 * 3D 2-point correlations

## Installation
nek-analyze depends on a number of other python packages.  Fortunately, they're all on PyPI, so they can be pip installed.
 - chest
 - mapcombine
 - Numpy
 - IPython (optional, for inter-node parallelism)
```
pip install numpy ipython mapcombine chest
```

## Usage

To use an existing analysis subpackage on a single node, simply:
```
usage: load.py [-h] [-f FRAME] [-e FRAME_END] [-s] [-c] [-n NINTERP] [-z] [-m]
               [-F] [-b] [-nb BLOCK] [-nt THREAD] [-d] [-p] [--series]
               [--mapreduce MAPREDUCE] [--post POST] [-v]
               [--params PARAM_PATH] [--chest CHEST_PATH] [--figs FIG_PATH]
               [--MR_init MR_INIT] [--reduce REDUCE] [--map MAP]
               [--single_pos]
               name

positional arguments:
  name                  Nek *.fld output file

optional arguments:
  -h, --help            show this help message and exit
  -f FRAME, --frame FRAME
                        [Starting] Frame number
  -e FRAME_END, --frame_end FRAME_END
                        Ending frame number
  -s, --slice           Display slice
  -c, --contour         Display contour
  -n NINTERP, --ninterp NINTERP
                        Interpolating order
  -z, --mixing_zone     Compute mixing zone width
  -m, --mixing_cdf      Plot CDF of box temps
  -F, --Fourier         Plot Fourier spectrum in x-y
  -b, --boxes           Compute box covering numbers
  -nb BLOCK, --block BLOCK
                        Number of elements to process at a time
  -nt THREAD, --thread THREAD
                        Number of threads to spawn
  -d, --display         Display plots with X
  -p, --parallel        Use parallel map (IPython)
  --series              Apply time-series analyses
  --mapreduce MAPREDUCE
                        Module containing Map and Reduce implementations
  --post POST           Module containing post_frame and post_series
  -v, --verbose         Should I be really verbose, that is: wordy?
  --params PARAM_PATH   Location of param file
  --chest CHEST_PATH    Location of chest directory
  --figs FIG_PATH       Location of figures
  --MR_init MR_INIT     MapReduce init function. Loaded from --mapreduce if
                        None.
  --reduce REDUCE       Reduce function. Loaded from --mapreduce if None.
  --map MAP             Map function. Loaded from --mapreduce if None.
  --single_pos          Position only in first output
```

## Extension

### Analysis (e.g. RTI, POD)
nek-analyze users need to populate four routines:
 * `Map`
 * `Reduce`
 * `post_frame`
 * `post_series`

