import warp as wp
import numpy as np
import polyscope as ps
import polyscope.imgui as gui
import os
from stretch import h, add_dx, PSViewer, Triplets
from mesh_complex import RodComplexBC, set_velocity_kernel, set_vx_kernel
from geometry.collision_cell import MeshCollisionDetector, collision_eps, stiffness
from geometry.static_scene import StaticScene

import os
from warp.sparse import bsr_set_from_triplets, bsr_zeros, bsr_mm, bsr_transposed, bsr_mv
from warp.optim.linear import cg
from scipy.linalg import solve, null_space
from scipy.sparse import block_diag
from scipy.io import loadmat
from g2m.viewer import MedialViewer
from g2m.dxdq import dxdq_jacobian
from g2m.collision_medial import MedialCollisionDetector
from g2m.medial import SlabMesh
from g2m.bary_centric import TetBaryCentricCompute
from g2m.nn import WarpEncoder
# from g2m.encoder import Encoder
# import torch
save_weight_only = False
# if enabled, only save the medial weights and quit
eps = 3e-3
def vec(t):
    return (t.T).reshape(-1)

@wp.kernel
def fill_U_triplets(mesh_id: int, offset: int, xcs: wp.array(dtype = wp.vec3), W: wp.array2d(dtype = float), triplets: Triplets):
    '''
    offset = mesh_id * xx for repeating meshes
    '''
    i, j, k = wp.tid()
    xx = W.shape[0]
    yy = W.shape[1]
    # block_nnz = 4 * xx * yy
    

    xid = i + offset
    idx = (xid * yy + j) * 4 + k #block_nnz * mesh_id
    triplets.rows[idx] = xid
    triplets.cols[idx] = j * 4 + k + mesh_id * yy * 4
    c = float(1.0)
    if k < 3:
        c = xcs[xid][k]
    triplets.vals[idx] = wp.diag(wp.vec3(W[i, j] * c))


class MedialRodComplex(RodComplexBC):
    def __init__(self, h, meshes=[], transforms=[], static_meshes = None):
        self.models = [mesh.split("/")[1].split(".")[0] for mesh in meshes]
        self.model_set = set(self.models)
        # model = meshes[0].split("/")[1].split(".")[0]
        d = []
        for model in self.model_set:
            Q = self.load_Q(model)
            d.append((model, Q))
        self.Q = dict(d)
            
        self.define_z(transforms)
        super().__init__(h, meshes, transforms, static_meshes)
        self.define_encoder()

        n_reduced = self.n_reduced
        self.A_reduced = np.zeros((n_reduced, n_reduced))
        self.b_reduced = np.zeros(n_reduced)
        self.define_U()
        self.compute_Um()

    def lbs_matrix(self, V, W):
        nvm = V.shape[0]
        v1 = np.ones((nvm, 4))
        v1[:, :3] = V
        lhs = np.hstack([W[:, j: j + 1] * v1 for j in range(W.shape[1])])
        return np.kron(lhs, np.identity(3))

    def define_U(self):
        self.U = np.zeros((self.n_nodes * 3, self.n_reduced))
        x0 = self.xcs.numpy()
        
        start = 0
        for i in range(self.n_meshes):
            Q = self.Q[self.models[i]]
            mesh_nodes = Q.shape[0]
            start_nxt = start + mesh_nodes
            xi = x0[start: start_nxt]
            Ui = self.lbs_matrix(xi, Q)
            self.U[start * 3: start_nxt * 3, i * self.n_modes: (i + 1) * self.n_modes] = Ui
            start = start_nxt
        
        self.wp_define_U()
        self.UwpT = bsr_transposed(self.Uwp)

    def wp_define_U(self):
        self.Uwp = bsr_zeros(self.n_nodes, self.n_modes // 3 * self.n_meshes, wp.mat33)
        triplets = Triplets()
        nnz = self.n_nodes * 4 * self.n_modes
        triplets.cols = wp.zeros((nnz, ), int)
        triplets.rows = wp.zeros((nnz, ), int)
        triplets.vals = wp.zeros((nnz,), wp.mat33)

        start = 0
        for i in range(self.n_meshes):
            Q = self.Q[self.models[i]]
            mesh_nodes = Q.shape[0]
            q = wp.array(Q, dtype = float)
            wp.launch(fill_U_triplets, (q.shape[0], q.shape[1], 4), inputs = [i, start, self.geo.xcs, q, triplets])
            start += mesh_nodes
        bsr_set_from_triplets(self.Uwp, triplets.rows, triplets.cols, triplets.vals, )

    def load_Q(self, model):
        Q = np.load(f"data/W_{model}.npy")
        Q = Q[:, :10]
        Q[:, 0] = 1.0
        return Q
        
    def define_z(self, transforms):
        Q = self.Q[self.models[0]]
        self.n_modes = Q.shape[1] * 12 
        self.n_meshes = len(transforms)

        self.n_reduced = self.n_modes * self.n_meshes
        self.z = np.zeros(self.n_reduced)
        for i in range(self.n_meshes):
            self.z[i * self.n_modes: i * self.n_modes + 9] = vec(np.identity(3))
        
        self.dz = np.zeros_like(self.z)


    def define_collider(self):
        super().define_collider()
        self.define_medials()
    
    def define_medials(self):        
        self.slabmeshes = dict([(model, SlabMesh(f"assets/{model}/ma/{model}.ma")) for model in self.model_set])
        
        
        R = np.zeros(0, float)
        E = np.zeros((0, 2), int)
        F = np.zeros((0, 3), int)
        V = np.zeros((0, 3))
        body = []

        cnt = 0
        for i in range(self.n_meshes):
            slabmesh = self.slabmeshes[self.models[i]]
            V0 = np.copy(slabmesh.V)
            v4 = np.ones((V0.shape[0], 4))
            v4[:, :3] = V0
            R0 = slabmesh.R
            E0 = slabmesh.E
            F0 = slabmesh.F

            Vi = (v4 @ self.transforms[i].T)[:, : 3]
            V = np.vstack([V, Vi])
            J3 = np.abs(np.linalg.det(self.transforms[i][:3, :3]))
            J = J3 ** (1./3.)
            # R = np.concatenate([R, np.copy(R0) * J])
            R = np.concatenate([R, np.copy(R0) * J])
            E = np.vstack((E, E0 + cnt))
            F = np.vstack((F, F0 + cnt))

            body += [i] * slabmesh.nv

            cnt += slabmesh.nv
        

        self.F_medial = F
        self.E_medial = E
        self.V_medial_rest = np.copy(V)
        self.V_medial = np.zeros_like(V)
        self.R_rest = np.copy(R)
        self.R = np.zeros_like(self.R_rest)
        self.body_medial = np.array(body, int)

        self.V_medial[:] = self.V_medial_rest
        self.R[:] = self.R_rest

        self.collider_medial = MedialCollisionDetector(
            self.V_medial, self.R_rest, self.E_medial, self.F_medial, self.body_medial, dense = False, ground = 0.0, static_objects = self.static_meshes)

        self.n_medial = self.V_medial.shape[0]

    def reset(self):
        if hasattr(self, "V_medial"):
            self.V_medial[:] = self.V_medial_rest[:]
            self.R[:] = self.R_rest
        super().reset()
        self.reset_z()

    def reset_z(self):
        t = self.transforms[0]
        self.z[:] = 0.0
        self.dz[:] = 0.0
        for i in range(self.n_meshes):
            self.z[self.n_modes * i: self.n_modes * i + 9] = vec(np.identity(3))

    def compute_Um(self):

        # jac = self.encoder.jacobian(x)
        diags = []
        start = 0
        for i in range(self.n_meshes):
            model = self.models[i]
            nv = self.slabmeshes[model].nv

            Vmi = self.V_medial_rest[start: start + nv]
            jaci = self.lbs_matrix(Vmi, self.W_medial[model])
            diags.append(jaci)
            start += nv
        self.Um = block_diag(diags, "csr")


    def line_search(self):
        alpha = super().line_search()
        self.z -= alpha * self.dz
        return alpha
    
    def solve(self):

        # self.A_reduced = self.U.T @ self.to_scipy_bsr() @ self.U
        # self.b_reduced = self.U.T @ self.b.numpy().reshape(-1)

        with wp.ScopedTimer("bsr mms"):
            tmp = bsr_mm(self.K_sparse, self.Uwp)
            self.A_reduced = bsr_mm(self.UwpT, tmp)

        with wp.ScopedTimer("bsr mv"):
            self.b_reduced = bsr_mv(self.UwpT, self.b)
        z_dim = self.dz.shape[0] // 3
        dz = wp.zeros((z_dim, ), dtype = wp.vec3)
        with wp.ScopedTimer("solve"):
            cg(self.A_reduced, self.b_reduced, dz, use_cuda_graph = True)
        dz = dz.numpy().reshape(-1)
        # dz = solve(self.A_reduced, self.b_reduced, assume_a="sym")
        self.dz[:] = dz
        self.states.dx.assign((self.U @ dz).reshape(-1, 3))
    # def line_search(self):
    #     self.z -= self.dz
    #     # wp.launch(add_dx, self.n_nodes, inputs=[self.states, 1.0])
    #     x = (self.U @ self.z).reshape((-1, 3))
    #     self.states.x.assign(x)
    
    #     return 1.0
    
    def define_encoder(self):
        W_medial_list = []
        for model in self.model_set:
            Q = self.Q[model]
            if os.path.exists(f"data/W_medial_{model}.npy") and Q.shape[0] > 300:
                weight = np.load(f"data/W_medial_{model}.npy")
            else: 
                with wp.ScopedTimer(f"define_encoder_{model}"):
                    intp = TetBaryCentricCompute(model)
                with wp.ScopedTimer(f"compute_weight_{model}"):
                    weight = intp.compute_weight(Q)
            W_medial_list.append((model, weight))
            if save_weight_only:
                np.save(f"data/W_medial_{model}.npy", weight)
                quit()
        self.W_medial = dict(W_medial_list)

    def define_K_sparse(self):
        if save_weight_only: 
            return
        else: 
            super().define_K_sparse()
    def get_VR(self):
        V = (self.Um @ self.z).reshape((-1, 3))# + self.V_medial_rest
        # V = self.V_medial_rest
        R = self.R_rest
        return V, R

    # def process_collision(self):
    #     # super().process_collision()
    #     # self.add_collision_to_sys_matrix()
    #     self.compute_A_reduced()

    #     V, R = self.get_VR()
    #     self.collider_medial.collision_set(V, R,)
    #     b, H, indices = self.collider_medial.analyze()

    #     # self.compute_Um()
    #     rhs = self.Um.T[:, indices] @ b
    #     A = self.Um.T[:, indices] @ H @ self.Um[indices, :]

    #     term = self.h * self.h * 2e4
    #     self.A_reduced += A * term
    #     self.col_b = rhs * term

    #     # self.add_collision_to_sys_matrix()

    def process_collision(self):
        with wp.ScopedTimer("collision"):
            with wp.ScopedTimer("detection"):
                self.n_pt, self.n_ee, self.n_ground = self.collider.collision_set("all") 
            with wp.ScopedTimer("hess & grad"):
                triplets = self.collider.analyze(self.b, self.n_pt, self.n_ee, self.n_ground)
                # triplets = self.collider.analyze(self.b)
            with wp.ScopedTimer("build_from_triplets"):
                self.add_collision_to_sys_matrix(triplets)

class PinnedWindMill(MedialRodComplex):
    def __init__(self, h, meshes=[], transforms=[], static_meshes = None):
        super().__init__(h, meshes, transforms, static_meshes)
        v_rst = self.xcs.numpy()
        x_rst = v_rst[:, 0]
        y_rst = v_rst[:, 1]
        pinned = (np.abs(x_rst) < eps) & (np.abs(y_rst) < eps)

        self.pinned = np.arange(self.n_nodes)[pinned]
        assert len(self.pinned) == 2

        y = v_rst[self.pinned]
        # pinned positions

        lhs = np.ones((2, 4), float)
        lhs[0, : 3] = y[0]
        lhs[1, : 3] = y[1]
        C = np.kron(lhs, np.identity(3))
        ns= null_space(C)
        self.ns = ns
        assert ns.shape == (12, 6)
        self.U_prime = np.zeros((self.n_reduced, self.n_reduced - 6))
        self.U_prime[:12, :6]= self.ns
        self.U_prime[12:, 6:] = np.identity(self.n_reduced - 12)
        self.UU_prime = self.U @ self.U_prime


    def solve(self):
        self.A_reduced = self.UU_prime.T @ self.to_scipy_bsr() @ self.UU_prime 
        self.b_reduced = self.UU_prime.T @ self.b.numpy().reshape(-1)

        dzprime = solve(self.A_reduced, self.b_reduced, assume_a="sym")
        self.dz[:] = self.U_prime @ dzprime
        self.states.dx.assign((self.U @ self.dz).reshape(-1, 3))
    
    def define_collider(self):
        self.collider = MeshCollisionDetector(self.states.x, self.T, self.indices, self.Bm, ground = None, static_objects = self.static_meshes)
        self.define_medials()
    # def process_collision(self):    
    #     pass

    # def compute_collision_energy(self):
    #     return 0.0
def bug_drop():
    n_meshes = 2
    meshes = ["assets/bug.tobj", "assets/tet.tobj"]
    # meshes = ["assets/bug.tobj"]

    transforms = [np.identity(4, dtype=float) for _ in range(n_meshes)]
    transforms[1][0, 3] = -2.0
    transforms[1][1, 3] = 1.
    transforms[0][1, 3] = 1.
    # rods = RodComplexBC(h, meshes, transforms)
    rods = MedialRodComplexDebug(h, meshes, transforms)
    # rods = EmbededMedialComplex(h, meshes, transforms)

    # viewer = PSViewer(rods)
    viewer = MedialViewer(rods)
    ps.set_user_callback(viewer.callback)
    ps.show()

def staggered_bug():
    model = "bunny"
    # model = "bug"
    n_meshes = 2
    # meshes = [f"assets/{model}/{model}.tobj"] * n_meshes
    meshes = [f"assets/bug/bug.tobj", f"assets/{model}/{model}.tobj"]
    transforms = [np.identity(4, dtype = float) for _ in range(n_meshes)]

    transforms[-1][:3, :3] = np.zeros((3, 3))
    transforms[-1][0, 1] = 1.5
    transforms[-1][1, 0] = 1.5
    transforms[-1][2, 2] = 1.5

    for i in range(n_meshes):
        # transforms[i][0, 3] = i * 0.5
        transforms[i][1, 3] = 1.2 + i * 0.25
        transforms[i][2, 3] = i * 1.2 - 0.8
    
    # rods = MedialRodComplex(h, meshes, transforms)

    # scale params for teapot
    static_meshes_file = ["assets/teapotContainer.obj"]
    scale = np.identity(4) * 3
    scale[3, 3] = 1.0

    # bouncy box
    static_meshes_file = ["assets/bouncybox.obj"]
    box_size = 4
    scale = np.identity(4) * box_size
    scale[3, 3] = 1.0
    scale[:3, 3] = np.array([0, box_size, box_size / 2], float)
    for i in range(n_meshes):
        transforms[i][1, 3] += box_size * 1.5
        
    
    static_bars = StaticScene(static_meshes_file, np.array([scale]))
    # static_bars = None
    rods = MedialRodComplex(h, meshes, transforms, static_bars)
    
    
    viewer = MedialViewer(rods, static_bars)
    ps.set_user_callback(viewer.callback)
    ps.show()


def windmill():
    # model = "bunny"
    model = "windmill"
    drop = "bunny"
    # model = "bug"
    n_meshes = 8
    # meshes = [f"assets/{model}/{model}.tobj"] * n_meshes
    meshes = [f"assets/{model}/{model}.tobj"] + [f"assets/{drop}/{drop}.tobj"] * (n_meshes - 1)
    transforms = [np.identity(4, dtype = float) for _ in range(n_meshes)]

    transforms[0][:3, :3] = np.zeros((3, 3))
    transforms[0][0, 0] = 0.5
    transforms[0][2, 1] = 0.5
    transforms[0][1, 2] = 0.5
    # transforms[-1][0, 1] = 1.5
    # transforms[-1][1, 0] = 1.5
    # transforms[-1][2, 2] = 1.5

    for i in range(1, n_meshes):
        transforms[i][0, 3] = 0.5 + i * 0.05
        transforms[i][1, 3] = i * 1.2
        transforms[i][2, 3] = i * 0.0
    
    # rods = MedialRodComplex(h, meshes, transforms)

    # scale params for teapot
    static_meshes_file = ["assets/teapotContainer.obj"]
    scale = np.identity(4) * 3
    scale[3, 3] = 1.0

    # bouncy box
    # static_meshes_file = ["assets/bouncybox.obj"]
    # box_size = 4
    # scale = np.identity(4) * box_size
    # scale[3, 3] = 1.0
    # scale[:3, 3] = np.array([0, box_size, box_size / 2], float)
    # for i in range(n_meshes):
    #     transforms[i][1, 3] += box_size * 1.5
        
    

    # static_bars = StaticScene(static_meshes_file, np.array([scale]))
    static_bars = None
    # rods = MedialRodComplex(h, meshes, transforms, static_bars)
    rods = PinnedWindMill(h, meshes, transforms, static_bars)
    
    
    viewer = MedialViewer(rods, static_bars)
    ps.set_user_callback(viewer.callback)
    ps.show()

if __name__ == "__main__":
    ps.init()
    ps.look_at((0, 4, 8), (0, 2, 0))
    # ps.set_ground_plane_mode("none")
    ps.set_ground_plane_height(-collision_eps)
    wp.config.max_unroll = 0
    wp.init()
    # bug_drop()
    # staggered_bug()
    windmill()
