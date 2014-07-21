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

## Usage

nek-analyze users need to populate four routines:
 * `Map`
 * `Reduce`
 * `post_frame`
 * `post_series`

