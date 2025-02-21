import numpy as np  
import warp as wp 
import igl
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import polyscope as ps
import polyscope.imgui as gui

class PSViewer: 
    def __init__(self, V, F, W):
        self.V = V
        self.F = F
        self.W = W

        self.ps_mesh = ps.register_surface_mesh("mesh", self.V, self.F)
        self.ui_handle = 0
        ps.set_user_callback(self.callback)

    def callback(self):
        changed, self.ui_handle = gui.SliderInt("handle", self.ui_handle, 0, self.W.shape[1] - 1)
        self.ps_mesh.add_scalar_quantity("BBW", self.W[:, self.ui_handle])
        
def test_bbw(): 
    V, T, F = igl.read_mesh("assets/hand.mesh")
    C, BE, _, _, _, _ = igl.read_tgf("assets/hand.tgf")

    Q = igl.read_dmat("assets/hand-pose.dmat")

    # print(v.shape, e.shape, p.shape, PE.shape)
    print(BE.shape, C.shape, Q.shape)
    _, b, bc = igl.boundary_conditions(V,T, C, np.zeros(0, dtype = int), BE, np.zeros((0, 0), dtype = int), np.zeros((0, 0), dtype = int))

    bbw = igl.BBW(2, 8)
    W = bbw.solve(V, T, b, bc)
    print(W.shape)
    return V, F, W

if __name__ == "__main__":
    ps.init()
    V, F, W = test_bbw()
    viewer = PSViewer(V, F, W)
    ps.show()
    # print(p[0])