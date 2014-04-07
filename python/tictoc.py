print_timers = False

""" Timers from SO """
def tic():
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc(label):
    import time
    if 'startTime_for_tictoc' in globals():
      if print_timers:
        print("    > " + str(time.time() - startTime_for_tictoc) + "s in " + label)
    else:
        print("Toc: start time not set")

