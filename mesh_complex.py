import warp as wp
from stretch import RodBCBase, PSViewer
from fem.interface import Rod, RodComplex
import polyscope as ps
import numpy as np

h = 1e-2
        
class RodComplexBC(RodBCBase, RodComplex):
    def __init__(self, h, meshes = [], transforms = []):
        self.meshes_filename = meshes 
        self.transforms = transforms
        super().__init__(h)


def multiple_drape():
    n_meshes = 3
    meshes = ["assets/bar2.tobj"] * n_meshes
    transforms = [np.identity(4, dtype = float) for _ in range(n_meshes)]
    for i in range(n_meshes):
        transforms[i][2, 3] = i * 2.0

    rods = RodComplexBC(h, meshes, transforms)
    viewer = PSViewer(rods)
    ps.set_user_callback(viewer.callback)
    ps.show()

if __name__ == "__main__":
    ps.init()
    wp.init()
    multiple_drape()
    # drape()