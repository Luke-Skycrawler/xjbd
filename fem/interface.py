import numpy as np
from .fem import SifakisFEM
from .geometry import RodGeometryGenerator, TOBJLoader
from .params import *


class Rod(SifakisFEM, TOBJLoader):
    def __init__(self, filename = default_tobj):
        self.filename = filename
        super().__init__()
        self.V = self.xcs.numpy()
        self.F = self.indices.numpy()
        self.mid = np.mean(self.V, axis = 0)
        self.V0 = self.V - self.mid
