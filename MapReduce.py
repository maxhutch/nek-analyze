
def merge(whole, part):
  whole['TMax']   = float(max(whole['TMax'],   part['TMax']))
  whole['TMin']   = float(min(whole['TMin'],   part['TMin']))
  whole['UAbs']   = float(max(whole['UAbs'],   part['UAbs']))
  whole['dx_max'] = float(max(whole['dx_max'], part['dx_max']))
  return 

