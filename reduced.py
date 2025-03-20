import warp as wp
from fem.interface import RodComplex
from stretch import RodBCBase, PSViewer, NewtonState, Triplets, h
from mesh_complex import RodComplexBC, init_transforms

import polyscope as ps
import numpy as np

from geometry.collision_cell import MeshCollisionDetector, collision_eps
from utils.tobj import import_tobj
from warp.sparse import bsr_axpy, bsr_set_from_triplets, bsr_zeros

class ReducedRodComplex(RodComplexBC):
    def __init__(self, h, meshes = [], transforms = []):
        super().__init__(h, meshes, transforms)
        
    
def reduced_bunny_rain():
    n_meshes = 10
    meshes = ["assets/bunny_5.tobj"] * n_meshes
    
    transforms = wp.zeros((n_meshes, ), dtype = wp.mat44)
    v, _ = import_tobj(meshes[0])
    bb_size = np.max(v, axis = 0) - np.min(v, axis = 0)
    wp.launch(init_transforms, (n_meshes,), inputs = [transforms, bb_size[0], bb_size[1], bb_size[2]])
    print(f"bb_size = {bb_size}")
    rods = RodComplexBC(h, meshes, transforms.numpy())
    viewer = PSViewer(rods)
    ps.set_user_callback(viewer.callback)
    ps.show()


if __name__ == "__main__":  
    ps.init() 
    ps.set_ground_plane_height(-collision_eps)
    wp.config.max_unroll = 0
    wp.init()
    
    reduced_bunny_rain()