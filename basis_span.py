import warp as wp 
import numpy as np
from utils.tobj import import_tobj
from scipy.linalg import solve
import igl
import polyscope as ps
from scipy.spatial.transform import Rotation as R
import polyscope.imgui as gui
model = "squishy_ball_lowlow"


def lbs_matrix(V, W):
    nvm = V.shape[0]
    v1 = np.ones((nvm, 4))
    v1[:, :3] = V
    lhs = np.hstack([W[:, j: j + 1] * v1 for j in range(W.shape[1])])
    return np.kron(lhs, np.identity(3))


class PSViewer:
    def __init__(self, x, v, F, x_target, U, z):
        self.target = ps.register_surface_mesh("x_target", x_target, F)
        self.target.add_scalar_quantity("indices", v[:, 0], enabled=True)
        self.fit = ps.register_surface_mesh("best fit", x, F)
        self.fit.add_scalar_quantity("indices", v[:, 0], enabled=True)
        self.ui_magnitude = 1.0
        ps.set_user_callback(self.callback)

        self.x, self.v= x, v
        
        self.U, self.z = U, z

        self.U0 = self.U[: , : 12]
        self.U_tilde = self.U[:, 12:]
        
        self.x0 = (self.U0 @ self.z[:12]).reshape((-1, 3)) + self.v
    def callback(self):
        changed, self.ui_magnitude = gui.SliderFloat("Magnitude", self.ui_magnitude, v_min = 0.0, v_max = 4)
        # x = self.ui_magnitude * (self.U @ self.z).reshape((-1, 3)) + self.v
        x = self.ui_magnitude * (self.U_tilde @ self.z[12:]).reshape((-1, 3)) + self.x0
        self.fit.update_vertex_positions(x)

def compute_ortho_loss():
    Q = np.load(f"data/W_{model}.npy")

    x_target = np.load(f"data/x_target.npy")
    
    r = R.from_rotvec([0, 0, np.pi/4])  # 45° around Z
    rot_matrix = r.as_matrix()

    x_target = x_target @ rot_matrix.T
    t = np.mean(x_target, axis = 0, keepdims= True)

    v, T = import_tobj(f"assets/squishyball/{model}.tobj")
    F = igl.boundary_facets(T)
    u = x_target - v

    U = lbs_matrix(v, Q[:, :])
    
    A = U.T @ U
    b = U.T @ u.reshape(-1)
    z = solve(A, b, assume_a="sym")
    x = (U @ z).reshape((-1, 3)) + v

    translated = v + t

    viewer = PSViewer(x, v, F, x_target, U, z)
if __name__ == "__main__":
    ps.init()
    compute_ortho_loss()
    ps.show()