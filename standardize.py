#!/usr/bin/env python3

# Grab input and output file names
from sys import argv
from nek import NekFile
in_file  = NekFile(argv[1])
out_file = NekFile(argv[2], in_file)

# load params from genrun.py input dictionary
import json
with open("{:s}.json".format(argv[3]), 'r') as f:
  params = json.load(f)

# convenience renaming
root = params["root_mesh"]
extent = params["extent_mesh"]
shape = params["shape_mesh"]

import numpy as np
for i in range(in_file.nelm):
  # load one element at a time
  n, rx, ru, rp, rt = in_file.get_elem(1)

  # x,y,z integer offsets 
  ix = int(0.5 + (rx[0,0,0] - root[0]) * shape[0] / (extent[0] - root[0]))
  iy = int(0.5 + (rx[0,0,1] - root[1]) * shape[1] / (extent[1] - root[1]))
  iz = int(0.5 + (rx[0,0,2] - root[2]) * shape[2] / (extent[2] - root[2]))

  # compute a standard element index
  j = int(ix + iy * shape[0] + iz * shape[0] * shape[1])
  if j < 0 or j >= in_file.nelm:
    print("Somethings wrong with ", j, rx[0,0,:])

  # write element in that position
  out_file.write(rx, ru, rp, rt, ielm = j)

# close up shop
in_file.close()
out_file.close()

