import numpy as np
from .fem import SifakisFEM
from .geometry import RodGeometryGenerator, TOBJLoader, TOBJComplex
from .params import *


class Rod(SifakisFEM, TOBJLoader):
    '''
    NOTE: need to have self.filename predefined before calling super().__init__()
    '''
    def __init__(self):
        if not hasattr(self, "filename"):
            self.filename = "assets/bar2.tobj"
        super().__init__()
        self.V = self.xcs.numpy()
        self.F = self.indices.numpy()
        self.mid = np.mean(self.V, axis = 0)
        self.V0 = self.V - self.mid

class RodComplex(SifakisFEM, TOBJComplex):
    '''
    NOTE: need to have self.meshes_filename and self.transforms predefined before calling super().__init__()
    '''
    def __init__(self):
        super().__init__()
        self.V = self.xcs.numpy()
        self.F = self.indices.numpy().reshape(-1, 3)
        self.mid = np.mean(self.V, axis = 0)
        self.V0 = self.V - self.mid

        