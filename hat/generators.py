class BaseGenerator(object):
    def __init__(self):
        pass
        
    """
    xs: list of ndarray
    ys: list of ndarray
    """
    def generate(self, xs, ys):
        raise NotImplementedError("generate() need to be implemented!")