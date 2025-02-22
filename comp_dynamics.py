import igl 
import numpy as np

import polyscope as ps


def load():
    
    V, _, _, F, _, _ = igl.read_obj("assets/elephant.obj")
    W = igl.read_dmat("assets/elephant-weights.dmat")
    C, BE, _, _, _, _ = igl.read_tgf("assets/elephant.tgf")

    T = igl.read_dmat("assets/elephant-anim.dmat")
    print(W.shape, T.shape, V.shape)

    return V, F, W, C, BE, T

class PSViewer:
    def __init__(self, V, F, W, T):
        self.V = V
        self.F = F
        self.W = W
        self.T = T

        self.M = igl.lbs_matrix(V, W)

        self.ps_mesh = ps.register_surface_mesh("mesh", self.V, self.F)
        ps.set_user_callback(self.callback)

        self.frame = 0
        self.n_frames = T.shape[1]

    def callback(self):

        Tf = self.T[:, self.frame].reshape(3, -1).T
        Vf = self.M @ Tf
        print(Tf[:4, :])
        # if (self.frame == 1):
        #     quit()
        self.ps_mesh.update_vertex_positions(Vf)
        
        self.frame += 1
        self.frame = self.frame % self.n_frames
        

if __name__ == "__main__":
    ps.init()
    V, F, W, C, BE, T = load()
    viewer = PSViewer(V, F, W, T)
    ps.show()
        

    # ps.show()