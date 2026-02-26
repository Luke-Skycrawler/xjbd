import numpy as np 
import warp as wp 

'''
implements [1]
reference:
[1]: Large Steps in Cloth Simulation
[2]: Dynamic Deformables: Implementation and Production Practicalities, SIGGRAPH course 2022
[3]: A Quadratic Bending Model for Inextensible Surfaces
'''

@wp.struct 
class ThinShell: 
    xcs: wp.array(dtype = wp.vec3)
    indices: wp.array(dtype = int)

@wp.kernel
def stretch_shear_kernel(x: wp.array(dtype = wp.vec3), geo: ThinShell, dudv: wp.array(dtype = wp.mat22), triplets: Triplets, b: wp.array(dtype = wp.vec3)):
    ej = wp.tid()
    e = ej // 9
    

class BW98ThinShell: 
    def __init__(self):
        super().__init__()
        
        self.b = wp.zeros((self.n_nodes, ), dtype = wp.vec3)
        self.dudv = wp.zeros((self.n_nodes, ), dtype = wp.mat22)
        self.geo = ThinShell()
        self.geo.xcs = self.xcs 
        self.geo.indices = self.indices

        self.define_K_sparse()

    def define_K_sparse(self): 
        pass 



