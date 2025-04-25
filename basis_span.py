import warp as wp 
import numpy as np
from utils.tobj import import_tobj
from scipy.linalg import solve
import igl
import polyscope as ps
model = "squishy_ball_lowlow"


def lbs_matrix(V, W):
    nvm = V.shape[0]
    v1 = np.ones((nvm, 4))
    v1[:, :3] = V
    lhs = np.hstack([W[:, j: j + 1] * v1 for j in range(W.shape[1])])
    return np.kron(lhs, np.identity(3))

def compute_ortho_loss():
    Q = np.load(f"data/W_{model}.npy")

    x_target = np.load(f"data/x_target.npy")
    t = np.mean(x_target, axis = 0, keepdims= True)

    v, T = import_tobj(f"assets/squishyball/{model}.tobj")
    F = igl.boundary_facets(T)
    u = x_target - v

    U = lbs_matrix(v, Q[:])
    
    A = U.T @ U
    b = U.T @ u.reshape(-1)
    z = solve(A, b, assume_a="sym")
    x = (U @ z).reshape((-1, 3)) + v

    translated = v + t

    ps.register_surface_mesh("x_target", x_target, F)
    ps.register_surface_mesh("best fit", x, F)
    ps.register_surface_mesh("translated", translated, F)

if __name__ == "__main__":
    ps.init()
    compute_ortho_loss()
    ps.show()