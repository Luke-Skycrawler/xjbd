import warp as wp
import numpy as np
import polyscope as ps
import polyscope.imgui as gui

from stretch import h, add_dx
from mesh_complex import RodComplexBC, set_velocity_kernel, set_vx_kernel
from geometry.collision_cell import MeshCollisionDetector, collision_eps, stiffness

import os
from scipy.linalg import solve
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

class MedialRodComplexDebug(RodComplexBC):
    def __init__(self, h, meshes=[], transforms=[]):
        self.load_Q()
        self.define_z(transforms)
        super().__init__(h, meshes, transforms)
        self.define_encoder()

        n_reduced = self.n_reduced
        self.A_reduced = np.zeros((n_reduced, n_reduced))
        self.b_reduced = np.zeros(n_reduced)
        self.define_U()
        self.Um = np.zeros((self.n_medial * 3, n_reduced))
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
        
    def load_Q(self):
        self.Q = np.load("data/W_bug.npy")
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
        self.slabmesh = SlabMesh("data/bug_v30.ma")
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

        # jac = self.encoder.jacobian(x)
        jac = self.lbs_matrix(self.V_medial_rest[:-2], self.W_medial)
        fill = jac
        # q6 = dxdq_jacobian(self.n_medial * 3 - 6, self.V_medial_rest[:-2])
        # self.Um[: -6, :6] = q6
        self.Um[: fill.shape[0], :fill.shape[1]] = fill

        self.Um[-6:, -6:] = np.identity(6)


    def add_collision_to_sys_matrix(self, triplets):
        super().add_collision_to_sys_matrix(triplets)
        self.compute_A_reduced()

    def compute_A_reduced(self):
        self.A_reduced = self.U.T @ self.to_scipy_bsr() @ self.U

    def compute_rhs(self):
        super().compute_rhs()
        b = self.b.numpy().reshape(-1)
        self.b_reduced = self.U.T @ b
        if ad_hoc:
            self.b_reduced += self.col_b

    def solve(self):
        dz = solve(self.A_reduced, self.b_reduced, assume_a="sym")
        self.dz[:] = dz
        self.states.dx.assign((self.U @ dz).reshape(-1, 3))

    def line_search(self):
        self.z -= self.dz
        # wp.launch(add_dx, self.n_nodes, inputs=[self.states, 1.0])
        x = (self.U @ self.z).reshape((-1, 3))
        self.states.x.assign(x)
    
        return 1.0

    def define_encoder(self):
        self.intp = TetBaryCentricCompute("bug", 30)
        self.W_medial = self.intp.compute_weight(self.Q)

    def get_VR(self):
        V = (self.Um @ self.z).reshape((-1, 3))# + self.V_medial_rest
        # V = self.V_medial_rest
        R = self.R_rest
        return V, R

    def process_collision(self):
        # super().process_collision()
        # self.add_collision_to_sys_matrix()
        self.compute_A_reduced()

        V, R = self.get_VR()
        self.collider_medial.collision_set(V, R,)
        b, H = self.collider_medial.analyze()

        self.compute_Um()
        rhs = self.Um.T @ b
        A = self.Um.T @ H @ self.Um

        term = self.h * self.h * 2e4
        self.A_reduced += A * term
        self.col_b = rhs * term

        # self.add_collision_to_sys_matrix()


class MedialRodComplex(MedialRodComplexDebug):
    def __init__(self, h, meshes=[], transforms=[]):
        super().__init__(h, meshes, transforms)
    
    def define_z(self, transforms):
        self.n_modes = self.Q.shape[1] * 12 
        self.n_meshes = len(transforms)

        self.n_reduced = self.n_modes * 2
        self.z = np.zeros(self.n_reduced)
        self.z[:9] = vec(np.identity(3))
        self.z[self.n_modes: self.n_modes + 9] = vec(np.identity(3))
        
        self.dz = np.zeros_like(self.z)

    def define_U(self):
        self.U = np.zeros((self.n_nodes * 3, self.n_reduced))
        nodes_per_mesh = self.n_nodes // self.n_meshes
        x0 = self.xcs.numpy()
        x00 = x0[:nodes_per_mesh]
        x01 = x0[nodes_per_mesh:]
        U0 = self.lbs_matrix(x00, self.Q)
        U1 = self.lbs_matrix(x01, self.Q)
        self.U[:nodes_per_mesh * 3, :self.n_modes] = U0
        self.U[nodes_per_mesh * 3:, self.n_modes:] = U1

    def define_collider(self):
        super().define_collider()
        self.slabmesh = SlabMesh("data/bug_v30.ma")
        V0 = np.copy(self.slabmesh.V)
        v4 = np.ones((V0.shape[0], 4))
        v4[:, :3] = V0
        R = self.slabmesh.R
        E = self.slabmesh.E

        if ad_hoc:
            V_bug0 = (v4 @ self.transforms[0].T)[:, :3]
            V_bug1 = (v4 @ self.transforms[1].T)[:, :3]

            self.cnt = V0.shape[0]
            V = np.vstack((V_bug0, V_bug1))
            R = np.concatenate((R, np.copy(R)))
            E = np.vstack((E, E + self.cnt))

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
        self.n_mdeial_per_mesh = self.n_medial // self.n_meshes

    def reset_z(self):
        t = self.transforms[0]
        self.z[:] = 0.0
        self.dz[:] = 0.0
        self.z[:9] = vec(np.identity(3))
        self.z[self.n_modes: self.n_modes + 9] = vec(np.identity(3))

    
    def compute_Um(self):

        # jac = self.encoder.jacobian(x)

        Vm_0 = self.V_medial_rest[:self.n_mdeial_per_mesh]
        Vm_1 = self.V_medial_rest[self.n_mdeial_per_mesh:]
        jac0 = self.lbs_matrix(Vm_0, self.W_medial)
        jac1 = self.lbs_matrix(Vm_1, self.W_medial)
        self.Um[: jac0.shape[0], :jac0.shape[1]] = jac0
        self.Um[self.n_mdeial_per_mesh * 3: self.n_mdeial_per_mesh * 3 + jac1.shape[0], self.n_modes:] = jac1

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
    n_meshes = 2 
    meshes = ["assets/bug.tobj"] * n_meshes
    # meshes = ["assets/bunny_5.tobj"] * n_meshes
    transforms = [np.identity(4, dtype = float) for _ in range(n_meshes)]
    transforms[1][:3, :3] = np.zeros((3, 3))
    transforms[1][0, 1] = 1
    transforms[1][1, 0] = 1
    transforms[1][2, 2] = 1

    for i in range(n_meshes):
        # transforms[i][0, 3] = i * 0.5
        transforms[i][1, 3] = 1.2 + i * 0.2
        transforms[i][2, 3] = i * 1.0
    
    rods = MedialRodComplex(h, meshes, transforms)
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
