import json
import numpy as np
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return ("ndarray", obj.tolist())
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)

class CustomDecoder(json.JSONDecoder):
    def decode(self, obj):
        if isinstance(obj, tuple) and obj[0] == "ndarray":
            return np.array(obj[1])
        # Let the base class default method raise the TypeError
        return json.JSONDecoder.decode(self, obj)
