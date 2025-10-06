import polyscope as ps 
import polyscope.imgui as gui 
import numpy as np
import warp as wp 
from warp.sparse import BsrMatrix
from fem.interface import Rod
from scipy.sparse import bsr_matrix, csr_matrix, bmat, diags
from scipy.sparse.linalg import eigsh
from scipy.linalg import null_space
from scipy.io import savemat, loadmat
from scipy.spatial.transform import Rotation as R
from warp.sparse import bsr_axpy, bsr_set_from_triplets, bsr_zeros
from fem.fem import Triplets
from fem.params import rho
from igl import lbs_matrix, massmatrix, dqs
import igl
import os
from g2m.viewer import MedialViewerInterface
from g2m.bary_centric import TetBaryCentricCompute
from g2m.naive_fitter import Fitter
# model = "bunny"
# model = "windmill"
model = "effel"
from stretch import eps
class PSViewer:
    def __init__(self, Q, V0, F):
        self.Q = Q
        self.V0 = V0
        self.F = F

        self.ui_deformed_mode = 0
        self.ui_dqs = False
        self.n_modes = Q.shape[1]
        self.B = lbs_matrix(self.V0, Q)
        self.T = np.zeros((4 * self.n_modes, 3))
        # self.T[:3, :] = np.eye(3)
        
        self.ui_magnitude = self.T[self.idx_to_T(self.ui_deformed_mode)]


        self.ui_Rx = 0.0
        self.ui_Ry = 0.0 
        self.ui_Rz = 0.0

        self.t = np.zeros((2, 3))
        self.q = np.zeros((2, 4))
        self.q[:, 3] = 1.0


        Q = np.zeros((self.Q.shape))
        Q[:] = self.Q[:] 
        # Q_max = np.max(np.abs(Q), axis = 0, keepdims = True)
        # Q /= Q_max
        
        Q_max_signed = np.max(Q, axis = 0, keepdims = True)
        Q_min = np.min(Q, axis = 0, keepdims = True)
        Q_range = Q_max_signed - Q_min
        Q[:, 1:] -= Q_min[:, 1:] 
        Q[:, 1:] /= Q_range[:, 1:]
        
        self.Q = Q

        self.ps_mesh = ps.register_surface_mesh("rod", V0, F)
        self.ps_mesh.add_scalar_quantity("weight", Q[:, 0], enabled = True)

    def current_magnitude(self):
        return self.T[self.idx_to_T(self.ui_deformed_mode)]
    def idx_to_T(self, idx):
        i = idx // 3
        j = idx % 3
        return i, j

    def to_dqs_weight(self, Q):
        return np.hstack([Q, 1.0 - Q])

    def compute_V(self):
        # return self.B @ self.T + self.V0

        rotation = R.from_euler('xyz', [self.ui_Rx, self.ui_Ry, self.ui_Rz], degrees = False)
        q = rotation.as_quat()
        self.q[self.ui_deformed_mode // 12] = q
        
        Q = self.to_dqs_weight(self.Q[:, self.ui_deformed_mode: self.ui_deformed_mode + 1])
        if self.ui_dqs:
            return dqs(self.V0.astype(float), Q, self.q, self.t)
        else: 
            rr = rotation.as_matrix().T
            rrt = np.vstack([rr, np.zeros((1, 3))])
            start = 0
            end = start + 4
            T = np.zeros((8, 3))
            T[start: end, :] = rrt
            T[start + 4: end + 3, :] = np.eye(3)
            B = lbs_matrix(self.V0, Q)
            return B @ T
        
    def callback(self):
        self.V_deform = self.compute_V()

        self.ps_mesh.update_vertex_positions(self.V_deform)

        changed, self.ui_deformed_mode = gui.InputInt("#mode", self.ui_deformed_mode, step = 1)
        if changed:
            self.ui_magnitude = self.T[self.idx_to_T(self.ui_deformed_mode)]
        changed, self.ui_dqs = gui.Checkbox("dqs", self.ui_dqs)
        changed, self.ui_magnitude = gui.SliderFloat("Magnitude", self.ui_magnitude, v_min = 0.0, v_max = 4)
        self.T[self.idx_to_T(self.ui_deformed_mode)] = self.ui_magnitude


        changed, self.ui_Rx = gui.SliderFloat("Rx", self.ui_Rx, v_min = -np.pi, v_max = np.pi)
        changed, self.ui_Ry = gui.SliderFloat("Ry", self.ui_Ry, v_min = -np.pi, v_max = np.pi)
        changed, self.ui_Rz = gui.SliderFloat("Rz", self.ui_Rz, v_min = -np.pi, v_max = np.pi)
        
        self.ps_mesh.add_scalar_quantity("weight", self.Q[:, self.ui_deformed_mode], enabled = True)

class PSViewerMedialSocket(MedialViewerInterface, PSViewer):
    def __init__(self, model, Q, V0, F):
        self.tbtt = TetBaryCentricCompute(model)

        self.V_rest = self.tbtt.slabmesh.V
        self.R_rest = self.tbtt.slabmesh.R
        self.E = self.tbtt.slabmesh.E
        V, T = self.tbtt.V, self.tbtt.T
        self.fitter = Fitter(V, T, self.tbtt.slabmesh)

        self.V_medial = self.fitter.fitted_V
        self.R = self.fitter.fitted_R
        
        super().__init__(Q, V0, F)

    def callback(self):
        super().callback()
        self.tbtt.deform(self.V_deform)
        V, R = self.fitter.V2p(self.V_deform)
        self.update_medial()
    
@wp.struct
class CSRTriplets:
    rows: wp.array(dtype = int)
    cols: wp.array(dtype = int) 
    vals: wp.array(dtype = float)

@wp.kernel
def compute_Hw(triplets: Triplets, triplets_Hw: CSRTriplets):
    i = wp.tid()
    ii = triplets.rows[i]
    jj = triplets.cols[i]
    mat = triplets.vals[i]

    triplets_Hw.rows[i] = ii
    triplets_Hw.cols[i] = jj
    triplets_Hw.vals[i] = wp.trace(mat)

    
class RodLBSWeight(Rod):
    def __init__(self):
        self.filename = f"assets/{model}/{model}.tobj"
        super().__init__()
        self.define_Hw()
        self.define_M()

    def define_M(self):
        V = self.xcs.numpy()
        T = self.T.numpy()
        # self.M is a vector composed of diagonal elements 
        self.Mnp = igl.massmatrix(V, T, igl.MASSMATRIX_TYPE_BARYCENTRIC).diagonal()
        M_diag = np.repeat(self.Mnp, 3)
        self.M_sparse = diags(M_diag)
        self.Mw = diags(self.Mnp * 3.0)

    def eigs_export(self, K, M):
        f = f"data/eigs/{model}.mat"
        savemat(f, {"K": K, "M": M}, long_field_names= True)
        print(f"exported matrices to {f}")

    def eigs(self):
        K = self.to_scipy_csr()
        # print("start weight space eigs")
        dim = K.shape[0]
        if dim >= 3000:
            self.eigs_export(K, self.Mw.tocsr())
            print("dimension exceeds scipy capability, switching to matlab")
            with wp.ScopedTimer("matlab eigs"):
                import matlab.engine
                eng = matlab.engine.start_matlab()
                eng.nullspace(model)
                data = loadmat(f"data/eigs/Q_{model}.mat")
                Q = data["Vv"].astype(np.float64)
                lam = data["D"].astype(np.float64)
                Q[:, 0] = 1.0
                # Q_norm = np.linalg.norm(Q, axis = 0, ord = np.inf, keepdims = True)
                # Q /= Q_norm
            # Q = loadmat(f"data/eigs/Q_{model}.mat")["Vv"].astype(np.float64)
            # lam = loadmat(f"data/eigs/Q_{model}.mat")["D"].astype(np.float64)
        else: 
            with wp.ScopedTimer("weight space eigs"):
                lam, Q = eigsh(K, k = 10, M = self.Mw, which = "SM")
                # lam, Q = eigsh(K, k = 10, which = "SM")
                # Q_norm = np.linalg.norm(Q, axis = 0, ord = np.inf, keepdims = True)
                # Q /= Q_norm
        return lam, Q
        
    def to_scipy_csr(self):
        ii = self.Hw.offsets.numpy()
        jj = self.Hw.columns.numpy()
        values = self.Hw.values.numpy()

        csr = csr_matrix((values, jj, ii), shape = (self.sys_dim, self.sys_dim))
        return csr
    
    def define_sys_dim(self):
        self.sys_dim = self.n_nodes

    def define_Hw(self):
        self.define_sys_dim()
        self.triplets_Hw = CSRTriplets()
        self.triplets_Hw.rows = wp.zeros((self.n_tets * 4 * 4,), dtype = int)
        self.triplets_Hw.cols = wp.zeros_like(self.triplets_Hw.rows)
        self.triplets_Hw.vals = wp.zeros((self.n_tets * 4 *4), dtype = float)
        
        self.Hw = bsr_zeros(self.sys_dim, self.sys_dim, float)
        wp.launch(compute_Hw, (self.n_tets * 4 * 4,), inputs = [self.triplets, self.triplets_Hw])
        bsr_set_from_triplets(self.Hw, self.triplets_Hw.rows, self.triplets_Hw.cols, self.triplets_Hw.vals)

class RodLBSWeightBC(RodLBSWeight):
    def __init__(self):
        super().__init__()
        self.define_M()
        self.define_Jw()

    def define_sys_dim(self):
        n_handles = 1
        self.sys_dim = self.n_nodes
        # self.sys_dim = self.n_nodes + 144 * n_handles
        # self.sys_dim = self.n_nodes + 1

    def compute_Aw(self):
        n = self.n_nodes

        A = np.zeros((3, 4, 3 * n, n))
        x_rst = self.xcs.numpy()
        xx = x_rst[:, 0]
        yy = x_rst[:, 1]
        zz = x_rst[:, 2]
        X = np.diag(xx)
        Y = np.diag(yy)
        Z = np.diag(zz)
        I = np.identity(n)

        Px = np.zeros((n * 3, n))
        Py = np.zeros_like(Px)
        Pz = np.zeros_like(Px)

        Px[:n, :] = np.identity(n, float)
        Py[n: 2 * n, :] = np.identity(n, float)
        Pz[2 * n: , :] = np.identity(n, float)

        Rs = [X, Y, Z, I]
        Ls = [Px, Py, Pz]
        for ii in range(3):
            for jj in range(4):
                A[ii, jj] = Ls[ii] @ Rs[jj]
        return A

    def get_contraint_weight(self):
        v_rst = self.xcs.numpy()        
        w = np.zeros((self.n_nodes, 3), float)

        x_rst = v_rst[:, 0]
        z_rst = v_rst[:, 2]
        
        # select = np.abs(x_rst) < eps * 10 
        # select2 = np.abs(z_rst) < eps * 10
        # select = select & select2
        select = np.array([80, 97])
        w[select[0], 0] = 1.0
        w[select[1], 1] = 1.0
        w[:, 2] = 1.0
        w[select, 2] = 0.0
        # w[x_rst < -0.5 + eps, 0] = 1.0
        # w[x_rst > 0.5 - eps, 1] = 1.0        
        # w[:, 0] = x_rst * 2
        # w[:, 1] = 1.
        return w

    def define_M(self):
        V = self.xcs.numpy()
        T = self.T.numpy()
        # self.M is a vector composed of diagonal elements 
        self.Mnp = igl.massmatrix(V, T, igl.MASSMATRIX_TYPE_BARYCENTRIC).diagonal() * rho
        M_diag = np.repeat(self.Mnp, 3)
        self.M_sparse = diags(M_diag)
        self.Mw = diags(self.Mnp * 3.0)

    def compute_J(self):
        w = self.get_contraint_weight()
        v_rst = self.xcs.numpy()        
        v1 = np.hstack((v_rst, np.ones((v_rst.shape[0], 1))))
        # v1 = np.hstack((np.ones((v_rst.shape[0], 1)), v_rst))
        # w = 1.0 - w
        # print(f"w sum = {w.sum()}")
        # v1 = v1 * w.reshape((-1, 1))

        # w2 = np.zeros_like(w)
        # w2[x_rst > 0.5 - eps] = 1.0
        js = np.hstack([v1 * w[:, i: i + 1] for i in range(w.shape[1])])
        # js = np.hstack([v1 * w.reshape((-1, 1)), v1 * w2.reshape((-1, 1))])
        # J = np.kron(np.identity(3, float), js)

        J = np.kron(np.identity(3, float), js)
        return J
        
    def define_Jw(self):
        J = self.compute_J()
        A = self.compute_Aw()
        Jwij = []
        for ii in range(3):
            for jj in range(4):
                Jwij.append(J.T @ A[ii, jj])
        # self.Jw = np.vstack(Jwij, )
        w = self.get_contraint_weight()
        self.Jw = w.T

        # w = np.zeros(self.n_nodes, float)
        # v_rst = self.xcs.numpy()
        # x_rst = v_rst[:, 0]
        # w[x_rst < -0.5 + eps] = 1.0
        # self.Jw = w.reshape(self.n_nodes, 1)
    
    def eigs_export(self, K, M):
        f = f"data/eigs/{self.filename}.mat"
        savemat(f, {"K": K, "M": M})
        print(f"exported matrices to {f}")

    def eigs(self):
        K = self.to_scipy_csr()
        M = self.Mw
        dim = K.shape[0]
        if dim >= 3000:
            self.eigs_export()
            print("dimension exceeds scipy capability, switching to matlab")
            with wp.ScopedTimer("matlab eigs"):
                import matlab.engine
                eng = matlab.engine.start_matlab()
                eng.nullspace()
                Q = loadmat(f"data/eigs/Q_{self.filename}.mat")["V"].astype(np.float64)
                lam = loadmat(f"data/eigs/Q_{self.filename}.mat")["D"].astype(np.float64)
                # Q_norm = np.linalg.norm(Q, axis = 0, ord = np.inf, keepdims = True)
                # Q /= Q_norm
        else: 
            na1 = null_space(self.Jw)
            print(f"na1 dim = {na1.shape}")
            tilde_K = na1.T @ K @ na1 
            tilde_M = na1.T @ M @ na1
            with wp.ScopedTimer("constrained weight space eigs"):
                lam, Ql = eigsh(tilde_K, k = 10, M = tilde_M, which = "SM")
                Ql = na1 @ Ql
                Q = Ql[:self.n_nodes]
                Q_norm = np.linalg.norm(Q, axis = 0, ord = np.inf, keepdims = True)
                Q /= Q_norm



        # v_rst = self.xcs.numpy()        
        # w = np.zeros((self.n_nodes, 1), float)
        # x_rst = v_rst[:, 0]
        # w[x_rst < -0.5 + eps, 0] = 1.0
        # w[x_rst > 0.5 - eps, 0] = 1.0
        # w = 1.0 - w

        # w = 1.0 - np.abs(self.get_contraint_weight())
        w = self.xcs.numpy()[:, 0:1]
        w1 = np.ones_like(w)
        v_rst = self.xcs.numpy()        
        x_rst = v_rst[:, 0]
        w[x_rst < -0.5 + eps, 0] = 0.0
        w[x_rst > 0.5 - eps, 0] = 0.0

        w1[x_rst < -0.5 + eps, 0] = 0.0
        w1[x_rst > 0.5 - eps, 0] = 0.0
        Q = np.hstack([w, w1, Q])
        # Q = np.hstack([w, Q])
        return lam, Q

        addon = block_array([[None, self.Jw.T], [self.Jw, None]], format = "csr")
        K += addon
        B_diags = np.zeros(self.sys_dim)
        B_diags[:self.n_nodes] = 1
        B_diags += 1e-7
        B = diags(B_diags)
        # print("start weight space eigs")
        with wp.ScopedTimer("constrained weight space eigs"):
            lam, Ql = eigsh(K, k = 10, M = B, which = "SM", tol = 1e-4)
            Q = Ql[:self.n_nodes]
            Q_norm = np.linalg.norm(Q, axis = 0, ord = np.inf, keepdims = True)
            Q /= Q_norm
        return lam, Q

def vis_weights(): 
    ps.init()
    ps.set_ground_plane_mode("none")
    wp.init()
    rod = RodLBSWeight()
    # rod = RodLBSWeightBC()
    lam, Q = None, None
    if not os.path.exists(f"data/W_{model}.npy"):
    # if True:
        lam, Q = rod.eigs()
        np.save(f"data/W_{model}.npy", Q)
    else:
        Q = np.load(f"data/W_{model}.npy")
    
    viewer = PSViewerMedialSocket(model, Q, rod.V0, rod.F)
    ps.set_user_callback(viewer.callback)
    ps.show()

if __name__ == "__main__":
    vis_weights()
    