#!/usr/bin/env python3

from sys import argv
from nek import NekFile
import numpy as np

ref = NekFile(argv[1])
test = NekFile(argv[2])

chunk = 1024
x_err, u_err, p_err, t_err = [0,0,0,0]

for i in range(int(ref.nelm / chunk)):
  n, rx, ru, rp, rt = ref.get_elem(chunk)
  n, tx, tu, tp, tt = test.get_elem(chunk)

  x_err += np.sum(np.square(rx - tx))
  u_err += np.sum(np.square(ru - tu))
  p_err += np.sum(np.square(rp - tp))
  t_err += np.sum(np.square(rt - tt))

x_err = np.sqrt(x_err/ref.ntot)
u_err = np.sqrt(u_err/ref.ntot)
p_err = np.sqrt(p_err/ref.ntot)
t_err = np.sqrt(t_err/ref.ntot)

print("X,U,P,T Errors: {:e} {:e} {:e} {:e}".format(x_err, u_err, p_err, t_err))

