class Struct:
    """Masquerade a dictionary with a structure-like behavior."""
    """From gprof2dot.py"""

    def __init__(self, attrs = None):
        if attrs is None:
            attrs = {}
        self.__dict__['_attrs'] = attrs
    
    def __getattr__(self, name):
        try:
            return self._attrs[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self._attrs[name] = value

    def __str__(self):
        return str(self._attrs)

    def __repr__(self):
        return repr(self._attrs)

