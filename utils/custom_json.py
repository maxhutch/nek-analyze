import json
import numpy as np
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {'__type__':"ndarray", 'val':obj.tolist()}
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)

class CustomDecoder(json.JSONDecoder):
    def __init__(self):
      json.JSONDecoder.__init__(self, object_hook=self.dict_to_object)

    def dict_to_object(self, d):
        if not '__type__' in d:
          return d
        if d['__type__'] == "ndarray":
          return np.array(d['val'])
        else:
          print("Don't know how to decode {:s}".format(d['__type__']))
        return d
