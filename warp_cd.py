import warp as wp 
import numpy as np 
from modal_warping import displace_u, ModalWarpingData
from fast_cd import PSViewer 
from fem.interface import Rod 
import polyscope as ps

model = "bar2"
class WarpViewer(PSViewer):
    def __init__(self, *args):
        super().__init__(*args)
        self.n_nodes = self.V0.shape[0]
        self.modal_w = ModalWarpingData()
        self.modal_w.W = wp.zeros((self.n_nodes), dtype = wp.vec3)
        self.modal_w.cnt = wp.zeros((self.n_nodes), dtype = int)
        self.modal_w.psi = wp.zeros_like(self.modal_w.W)
        self.modal_w.u = wp.zeros_like(self.modal_w.W)
        psi = np.outer(self.Q[:, 0], np.array([[np.pi / 6.0, 0.0, 0.0]]))
        self.modal_w.psi.assign(psi)

    def compute_V_deform(self):
        
        zn = wp.zeros_like(self.modal_w.W)
        zn.assign(self.V0)
        wp.launch(displace_u, dim = (self.n_nodes, ), inputs = [self.modal_w, zn])
        self.V_deform[:] = self.V0 + self.modal_w.u.numpy()

    
def vis_warp_cd():
    ps.init()
    ps.set_ground_plane_mode("none")
    wp.init()
    rod = Rod()
    Q = np.load(f"data/W_{model}.npy")


    viewer = WarpViewer(Q, rod.V0, rod.F)
    ps.set_user_callback(viewer.callback)
    ps.show()

if __name__ == "__main__":
    vis_warp_cd()