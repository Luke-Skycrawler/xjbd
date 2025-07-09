import warp as wp 
import numpy as np
from utils.tobj import import_tobj
from scipy.linalg import solve
import igl
import polyscope as ps
from scipy.spatial.transform import Rotation as R
import polyscope.imgui as gui
from modal_warping import ModalWarpingRod, MWViewer
model = "squishy"


def lbs_matrix(V, W):
    nvm = V.shape[0]
    v1 = np.ones((nvm, 4))
    v1[:, :3] = V
    lhs = np.hstack([W[:, j: j + 1] * v1 for j in range(W.shape[1])])
    return np.kron(lhs, np.identity(3))

class WarpedRod(ModalWarpingRod):
    def __init__(self, filename = f"assets/squishy/squishy.tobj"):
        super().__init__(filename)

    def compute_Q(self):

        Q = np.load(f"data/W_{model}.npy")
        Q[:, 0] = 1.0
        V = self.V0
        self.Q = lbs_matrix(V, Q)
        

class PSViewer:
    def __init__(self, x, v, F, x_target, U, z):
        # self.target = ps.register_surface_mesh("x_target", x_target, F)
        # self.target.add_scalar_quantity("indices", v[:, 0], enabled=True)
        self.fit = ps.register_surface_mesh("best fit", x, F)
        self.fit.add_scalar_quantity("indices", v[:, 0], enabled=True)
        self.ui_magnitude = 1.0
        ps.set_user_callback(self.callback)

        self.x, self.v= x, v
        
        self.U, self.z = U, z

        self.U0 = self.U[: , : 12]
        self.U_tilde = self.U[:, 12:]
        
        # self.x0 = (self.U0 @ self.z[:12]).reshape((-1, 3)) + self.v
        self.x0 = self.v
        self.rod = WarpedRod()
        self.x_target = x_target
        self.ui_use_modal_warping = True
    def callback(self):
        changed, self.ui_magnitude = gui.SliderFloat("Magnitude", self.ui_magnitude, v_min = 0.0, v_max = 10.0)
        changed, self.ui_use_modal_warping = gui.Checkbox("Use Modal Warping", self.ui_use_modal_warping)
        # x = self.ui_magnitude * (self.U @ self.z).reshape((-1, 3)) + self.v

        r = R.from_rotvec([0, 0, np.pi/18])  # 45° around Z
        rot_matrix = r.as_matrix()

        # x_disp = self.x0 @ rot_matrix.T
        # Qi= (x_disp - self.x0).reshape(-1)

        # Qi = self.x_target - (self.x0 + np.array([[0., 0.7, -0.4]]))

        # Qi = self.U0 @ self.z[:12]
        Qi = (self.U_tilde @ self.z[12:])


        if self.ui_use_modal_warping:
            disp = self.rod.compute_Psi(Qi, self.ui_magnitude)
            x = self.x0 + disp.reshape((-1, 3))
        else: 
            x = self.ui_magnitude * Qi.reshape((-1, 3)) + self.x0

        self.fit.update_vertex_positions(x)

def vec(t):
    return (t.T).reshape(-1)

def compute_ortho_loss():
    Q = np.load(f"data/W_{model}.npy")
    Q[:, 0] = 1.0

    x_target = np.load(f"data/x_target.npy")
    
    # r = R.from_rotvec([0, 0, np.pi/4])  # 45° around Z
    # rot_matrix = r.as_matrix()

    # x_target = x_target @ rot_matrix.T
    t = np.mean(x_target, axis = 0, keepdims= True)

    v, T = import_tobj(f"assets/{model}/{model}.tobj")
    F = igl.boundary_facets(T)
    u = x_target - v

    U = lbs_matrix(v, Q[:, :])
    
    A = U.T @ U
    b = U.T @ u.reshape(-1)
    # z = solve(A, b, assume_a="sym")
    zs0 = vec(np.array(
        [[0.0, 0.5, 0.0], 
         [0.5, 0.0, 0.0], 
         [0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0]]))
    za0 = vec(np.array(
        [[0.0, -0.5, 0.0], 
         [0.5, 0.0, 0.0], 
         [0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0]]))
    z = np.zeros_like(b)
    z[24:36] = zs0 * 1e-2
    
    x = (U @ z).reshape((-1, 3)) + v

    translated = v + t

    viewer = PSViewer(x, v, F, x_target, U, z)
    
def view_mw():
    wp.init()
    ps.init()
    ps.set_ground_plane_mode("none")
    rod = WarpedRod()
    viewer = MWViewer(rod)

    ps.set_user_callback(viewer.callback)
    ps.show()

if __name__ == "__main__":
    # view_mw()
    ps.init()
    ps.set_ground_plane_mode("none")
    compute_ortho_loss()
    ps.show()