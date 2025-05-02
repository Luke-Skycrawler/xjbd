import warp as wp
import numpy as np
import polyscope as ps
import polyscope.imgui as gui

from stretch import h, add_dx, PSViewer, Triplets
from mesh_complex import RodComplexBC, set_velocity_kernel, set_vx_kernel
from geometry.collision_cell import MeshCollisionDetector, collision_eps, stiffness
from geometry.static_scene import StaticScene

import os
from warp.sparse import bsr_set_from_triplets, bsr_zeros, bsr_mm, bsr_transposed, bsr_mv
from warp.optim.linear import cg
from scipy.linalg import solve
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

ad_hoc = True

def vec(t):
    return (t.T).reshape(-1)
@wp.kernel
def fill_U_triplets(mesh_id: int, xcs: wp.array(dtype = wp.vec3), W: wp.array2d(dtype = float), triplets: Triplets):
    i, j, k = wp.tid()
    xx = W.shape[0]
    yy = W.shape[1]
    block_nnz = 4 * xx * yy

    idx = (i * yy + j) * 4 + k + block_nnz * mesh_id
    xid = i + mesh_id * xx
    triplets.rows[idx] = xid
    triplets.cols[idx] = j * 4 + k + mesh_id * yy * 4
    c = float(1.0)
    if k < 3:
        c = xcs[xid][k]
    triplets.vals[idx] = wp.diag(wp.vec3(W[i, j] * c))


class MedialRodComplexDebug(RodComplexBC):
    def __init__(self, h, meshes=[], transforms=[], static_meshes = None):
        model = meshes[0].split("/")[1].split(".")[0]
        self.load_Q(model)
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
        self.U[:-12, : -12] = self.lbs_matrix(self.xcs.numpy()[:-4], self.Q)
        self.U[-12:, -12:] = np.identity(12)
        
    def load_Q(self, model):
        # Q = np.load("data/W_bug.npy")
        Q = np.load(f"data/W_{model}.npy")
        self.Q = Q[:, :]
        self.Q[:, 0] = 1.0
        
        
    def define_z(self, transforms):
        t = np.array(transforms)
        t = t[0, :3, :4]
        self.n_reduced = self.Q.shape[1] * 12 + 12
        self.z = np.zeros(self.n_reduced)
        self.z[:9] = vec(np.identity(3))
        
        self.dz = np.zeros_like(self.z)

    def define_collider(self):
        super().define_collider()
        # self.slabmesh = SlabMesh("data/bug_v30.ma")
        self.define_medials()

    def define_medials(self):
        model = self.meshes_filename[0].split("/")[1].split(".")[0]
        assert model in ["bug", "squishy", "bunny"]
        self.slabmesh = SlabMesh(f"assets/{model}/ma/{model}.ma")
        V = np.copy(self.slabmesh.V)
        v4 = np.ones((V.shape[0], 4))
        v4[:, :3] = V
        V = (v4 @ self.transforms[0].T)[:, :3]
        R = self.slabmesh.R
        E = self.slabmesh.E

        if ad_hoc:
            self.cnt = V.shape[0]
            V = np.vstack((V, self.xcs.numpy()[-2:, :].reshape((-1, 3))))
            R = np.concatenate((R, np.array([0.1, 0.1])))
            E = np.vstack((E, np.array([[self.cnt, self.cnt + 1]])))

        self.E_medial = E
        self.V_medial_rest = np.copy(V)
        self.V_medial = np.zeros_like(V)
        self.R_rest = np.copy(R)
        self.R = np.zeros_like(self.R_rest)

        self.V_medial[:] = self.V_medial_rest
        self.R[:] = self.R_rest

        self.collider_medial = MedialCollisionDetector(
            self.V_medial, self.R_rest, self.E_medial, self.slabmesh.F)

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
        self.z[:9] = vec(np.identity(3))
        self.z[-12:] = self.xcs.numpy()[-4:].reshape(-1)

    def compute_Um(self):
        self.Um = np.zeros((self.n_medial * 3, self.n_reduced))
        # jac = self.encoder.jacobian(x)
        jac = self.lbs_matrix(self.V_medial_rest[:-2], self.W_medial)
        fill = jac
        # q6 = dxdq_jacobian(self.n_medial * 3 - 6, self.V_medial_rest[:-2])
        # self.Um[: -6, :6] = q6
        self.Um[: fill.shape[0], :fill.shape[1]] = fill

        self.Um[-6:, -6:] = np.identity(6)


    # def add_collision_to_sys_matrix(self, triplets):
    #     super().add_collision_to_sys_matrix(triplets)
    #     self.compute_A_reduced()

    def compute_A_reduced(self):
        self.A_reduced = self.U.T @ self.to_scipy_bsr() @ self.U

    # def compute_rhs(self):
    #     super().compute_rhs()
    #     b = self.b.numpy().reshape(-1)
    #     self.b_reduced = self.U.T @ b
    #     if ad_hoc:
    #         self.b_reduced += self.col_b

    def solve(self):

        self.A_reduced = self.U.T @ self.to_scipy_bsr() @ self.U
        self.b_reduced = self.U.T @ self.b.numpy().reshape(-1)
        
        dz = solve(self.A_reduced, self.b_reduced, assume_a="sym")
        self.dz[:] = dz
        self.states.dx.assign((self.U @ dz).reshape(-1, 3))

    # def line_search(self):
    #     self.z -= self.dz
    #     # wp.launch(add_dx, self.n_nodes, inputs=[self.states, 1.0])
    #     x = (self.U @ self.z).reshape((-1, 3))
    #     self.states.x.assign(x)
    
    #     return 1.0

    def define_encoder(self):
        # self.intp = TetBaryCentricCompute("bug", 30)
        model = self.meshes_filename[0].split("/")[1].split(".")[0]
        self.intp = TetBaryCentricCompute(model)
        self.W_medial = self.intp.compute_weight(self.Q)

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


class MedialRodComplex(MedialRodComplexDebug):
    def __init__(self, h, meshes=[], transforms=[], static_meshes = None):
        super().__init__(h, meshes, transforms, static_meshes)
    
    def define_z(self, transforms):
        self.n_modes = self.Q.shape[1] * 12 
        self.n_meshes = len(transforms)

        self.n_reduced = self.n_modes * self.n_meshes
        self.z = np.zeros(self.n_reduced)
        self.z[:9] = vec(np.identity(3))
        self.z[self.n_modes: self.n_modes + 9] = vec(np.identity(3))
        
        self.dz = np.zeros_like(self.z)

    def define_U(self):
        self.U = np.zeros((self.n_nodes * 3, self.n_reduced))
        nodes_per_mesh = self.n_nodes // self.n_meshes
        x0 = self.xcs.numpy()
        
        for i in range(self.n_meshes):
            xi = x0[i * nodes_per_mesh: (i + 1) * nodes_per_mesh]
            Ui = self.lbs_matrix(xi, self.Q)
            self.U[i * nodes_per_mesh * 3: (i + 1) * nodes_per_mesh * 3, i * self.n_modes: (i + 1) * self.n_modes] = Ui
            
        self.Uwp = bsr_zeros(self.n_nodes, self.n_modes // 3 * self.n_meshes, wp.mat33)
        q = wp.array(self.Q, dtype = float)
        triplets = Triplets()
        nnz = q.shape[0] * q.shape[1] * 4 * self.n_meshes
        triplets.cols = wp.zeros((nnz, ), int)
        triplets.rows = wp.zeros((nnz, ), int)
        triplets.vals = wp.zeros((nnz,), wp.mat33)

        for i in range(self.n_meshes):
            wp.launch(fill_U_triplets, (q.shape[0], q.shape[1], 4), inputs = [i, self.geo.xcs, q, triplets])
        bsr_set_from_triplets(self.Uwp, triplets.rows, triplets.cols, triplets.vals, )
        self.UwpT = bsr_transposed(self.Uwp)


        # with wp.ScopedTimer("bsr mms"):
        #     tmp = bsr_mm(self.K_sparse, self.Uwp)
        #     self.A_reduced = bsr_mm(self.UwpT, tmp)

    def define_medials(self):
        model = self.meshes_filename[0].split("/")[1].split(".")[0]
        assert model in ["bug", "squishy", "bunny"]
        self.slabmesh = SlabMesh(f"assets/{model}/ma/{model}.ma")
        V0 = np.copy(self.slabmesh.V)
        v4 = np.ones((V0.shape[0], 4))
        v4[:, :3] = V0
        R0 = self.slabmesh.R
        E0 = self.slabmesh.E
        F0 = self.slabmesh.F
        
        R = np.zeros(0, float)
        E = np.zeros((0, 2), int)
        F = np.zeros((0, 3), int)
        V = np.zeros((0, 3))
        self.n_mdeial_per_mesh = V0.shape[0]

        for i in range(self.n_meshes):
            Vi = (v4 @ self.transforms[i].T)[:, : 3]
            cnt = i * self.n_mdeial_per_mesh
            V = np.vstack([V, Vi])
            J3 = np.linalg.det(self.transforms[i][:3, :3])
            J = np.abs(np.power(J3, 1 / 3))
            # R = np.concatenate([R, np.copy(R0) * J])
            R = np.concatenate([R, np.copy(R0)])
            E = np.vstack((E, E0 + cnt))
            F = np.vstack((F, F0 + cnt))

        self.F_medial = F
        self.E_medial = E
        self.V_medial_rest = np.copy(V)
        self.V_medial = np.zeros_like(V)
        self.R_rest = np.copy(R)
        self.R = np.zeros_like(self.R_rest)

        self.V_medial[:] = self.V_medial_rest
        self.R[:] = self.R_rest

        self.collider_medial = MedialCollisionDetector(
            self.V_medial, self.R_rest, self.E_medial, self.F_medial, ground = 0.0, static_objects = self.static_meshes)

        self.n_medial = self.V_medial.shape[0]

    def reset_z(self):
        t = self.transforms[0]
        self.z[:] = 0.0
        self.dz[:] = 0.0
        self.z[:9] = vec(np.identity(3))
        self.z[self.n_modes: self.n_modes + 9] = vec(np.identity(3))

    
    def compute_Um(self):

        # jac = self.encoder.jacobian(x)
        diags = []
        for i in range(self.n_meshes):
            Vmi = self.V_medial_rest[i * self.n_mdeial_per_mesh: (i + 1) * self.n_mdeial_per_mesh]
            jaci = self.lbs_matrix(Vmi, self.W_medial)
            diags.append(jaci)
            # self.Um[i * self.n_mdeial_per_mesh * 3: (i) * self.n_mdeial_per_mesh * 3 + jaci.shape[0], i * self.n_modes: (i + 1) * self.n_modes] = jaci
        self.Um = block_diag(diags, "csr")

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
    

# def staggered_bug():
#     n_meshes = 2 
#     meshes = ["assets/bug.tobj"] * n_meshes
#     # meshes = ["assets/bunny_5.tobj"] * n_meshes
#     transforms = [np.identity(4, dtype = float) for _ in range(n_meshes)]
#     transforms[1][:3, :3] = np.zeros((3, 3))
#     transforms[1][0, 1] = 1
#     transforms[1][1, 0] = 1
#     transforms[1][2, 2] = 1

#     for i in range(n_meshes):
#         # transforms[i][0, 3] = i * 0.5
#         transforms[i][1, 3] = 1.2 + i * 0.2
#         transforms[i][2, 3] = i * 1.0
    
#     rods = MedialRodComplex(h, meshes, transforms)
#     viewer = MedialViewer(rods)
#     ps.set_user_callback(viewer.callback)
#     ps.show()

def staggered_bug():
    
    n_meshes = 2
    # meshes = ["assets/bug.tobj"] * n_meshes
    meshes = ["assets/squishy/squishy.tobj"] * n_meshes
    # meshes = ["assets/bunny_5.tobj"] * n_meshes
    transforms = [np.identity(4, dtype = float) for _ in range(n_meshes)]
    transforms[1][:3, :3] = np.zeros((3, 3))
    transforms[1][0, 1] = 1
    transforms[1][1, 0] = 1
    transforms[1][2, 2] = 1

    for i in range(n_meshes):
        # transforms[i][0, 3] = i * 0.5
        # transforms[i][1, 3] = 1.2 + i * 0.2
        transforms[i][1, 3] = 1.2 + i * 0.25
        transforms[i][2, 3] = i * 1.2 - 0.4
    
    # rods = MedialRodComplex(h, meshes, transforms)
    static_meshes_file = ["assets/teapotContainer.obj"]
    scale = np.identity(4) * 3
    scale[3, 3] = 1.0
    static_bars = StaticScene(static_meshes_file, np.array([scale]))
    static_bars = None
    rods = MedialRodComplex(h, meshes, transforms) #, static_bars)
    
    
    viewer = MedialViewer(rods)
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
    staggered_bug()
