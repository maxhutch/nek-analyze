def MR_init(args, params):
  """ Initialize MapReduce data """
  from nek import NekFile

  # Open input and output Nek files
  input_file = NekFile(args.fname)
  output_file = NekFile("{:s}.rpkg".format(args.fname), input_file)

  # return a cleaned up version of locals
  ans = locals()
  del ans['args']
  del ans['NekFile']

  return ans

def map_(pos, vel, p, t, params, ans = None):
  """ Map operations onto chunk of elements """

  # Nicer names
  root   = params["root_mesh"]
  extent = params["extent_mesh"]
  shape  = params["shape_mesh"]

  # Loop over chunk of elements elements
  for ielm in range(pos.shape[2]):
    ix = int(0.5 + (pos[0,0,ielm] - root[0]) * shape[0] / (extent[0] - root[0]))
    iy = int(0.5 + (pos[0,1,ielm] - root[1]) * shape[1] / (extent[1] - root[1]))
    iz = int(0.5 + (pos[0,2,ielm] - root[2]) * shape[2] / (extent[2] - root[2]))

    # Compute standard global index
    jelm = int(ix + iy * shape[0] + iz * shape[0] * shape[1])

    # write element in that position
    """
    ans["output_file"].write(pos[:,:,ielm:ielm+1], 
                             vel[:,:,ielm:ielm+1], 
                             p[:,ielm:ielm+1], 
                             t[:,ielm:ielm+1], 
                             ielm = jelm)
    """

  return

def reduce_(whole, part):
  """ Reduce results into a single output object (dict) """

  # Close the files
  part["output_file"].close()
  part["input_file"].close()

  return 

