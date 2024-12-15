import polyscope as ps
import polyscope.imgui as gui
import numpy as np
import warp as wp
from fem.interface import Rod
from fem.params import model

class PSViewer:
    def __init__(self, Q, V0, F):
        self.Q = Q
        self.V0 = V0
        self.F = F
        self.magnitude = 1.0
        self.ps_mesh = ps.register_surface_mesh("rod", V0, F)

        self.ui_deformed_mode = 0

        self.ui_magnitude = 1.0
    def callback(self):
        Qi = self.Q[:, self.ui_deformed_mode]

        disp = self.ui_magnitude * Qi 
        disp = disp.reshape((-1, 3))

        self.V_deform = self.V0 + disp 

        self.ps_mesh.update_vertex_positions(self.V_deform)

        changed, self.ui_deformed_mode = gui.InputInt("#mode", self.ui_deformed_mode, step = 1)

        changed, self.ui_magnitude = gui.SliderFloat("Magnitude", self.ui_magnitude, v_min = 0.0, v_max = 5.0)


def vis_eigs():
    ps.init()
    ps.set_ground_plane_mode("none")
    wp.init()
    rod = Rod()
    lam, Q = None, None
    # if not os.path.exists(f"Q_{model}.npy"):
    if True:
        lam, Q = rod.eigs()
    # if True:
    #     K = rod.K
    #     savemat(f"K_{model}.mat", {"K": K})
    #     quit()
    else:
        Q = np.load(f"Q_{model}.npy")

    mid, V0, F = rod.mid, rod.V0, rod.F

    viewer = PSViewer(Q, V0, F)
    # np.save(f"Q_{model}.npy", Q)
    ps.set_user_callback(viewer.callback)
    ps.show()

if __name__ == "__main__":
    # vis_eigs_KKT()
    vis_eigs()
