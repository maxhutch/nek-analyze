print_timers = True

""" Timers from SO """
def tic():
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc(label):
    import time
    if 'startTime_for_tictoc' in globals():
      if print_timers:
        print("    > {:f}s in {:s}".format(time.time() - startTime_for_tictoc, label))
    else:
        print("Toc: start time not set")

