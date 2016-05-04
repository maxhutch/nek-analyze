def MR_init(args, params, frame):
  """ Initialize MapReduce data """

  # return a cleaned up version of locals
  ans = locals()
  del ans['args']

  return ans


def map_(pos, vel, p, t, params, scratch = None, last=False):
  """ Map operations onto chunk of elements """

  return

def reduce_(whole, part):
  """ Reduce results into a single output object (dict) """

  return 

