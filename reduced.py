import warp as wp
from fem.interface import RodComplex
from stretch import RodBCBase, NewtonState, Triplets, h, add_dx, PSViewer
from mesh_complex import RodComplexBC, init_transforms, init_velocities

import polyscope as ps
import polyscope.imgui as gui
import numpy as np

from geometry.collision_cell import MeshCollisionDetector, collision_eps
from utils.tobj import import_tobj
from warp.sparse import bsr_axpy, bsr_set_from_triplets, bsr_zeros, bsr_mm, bsr_transposed, bsr_mv
from fast_cd import RodLBSWeight
import os 
from scipy.linalg import solve
from scipy.io import loadmat
from igl import lbs_matrix
from fast_cd import CSRTriplets, compute_Hw
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from fast_cd import model

def dxdq_jacobian(n_nodesx3, V):
    n_nodes = n_nodesx3 // 3
    q6 = np.zeros((n_nodesx3, 6))
    skew = lambda x: np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
    print(f"n_nodes: {n_nodes}")
    for i in range(n_nodes):
        q6[3 * i: 3 * i + 3, : 3] = np.eye(3)
    for i in range(V.shape[0]):
        q6[3 * i: 3 * i + 3, 3:] = skew(V[i])
    return q6

@wp.kernel
def init_spin(states: NewtonState):
    i = wp.tid()
    axis = wp.vec3(0., 1., 0.)
    pi = states.x[i]
    xdot = wp.cross(axis, pi)
    states.xdot[i] = xdot

class ReducedRodComplex(RodComplexBC):
    def __init__(self, h, meshes = [], transforms = []) :
        super().__init__(h, meshes, transforms)

        self.define_U()
        n_reduced = self.n_reduced
        self.A_reduced = np.zeros((n_reduced, n_reduced))
        self.b_reduced = np.zeros(n_reduced)

    def define_U(self):

        Q = np.load(f"data/W_{model}.npy")

        print(f"Q = {Q.shape}, Q.variance = {np.var(Q)}, Q.mean = {np.mean(Q)}")
        
        self.Q = Q
        self.n_meshes = len(self.meshes_filename)
        self.n_modes = Q.shape[1] * 12

        self.n_reduced = self.n_modes * self.n_meshes
        self.U = np.zeros((self.n_nodes * 3, self.n_reduced))
        nodes_per_mesh = self.n_nodes // self.n_meshes
        x0 = self.xcs.numpy()
        
        for i in range(self.n_meshes):
            xi = x0[i * nodes_per_mesh: (i + 1) * nodes_per_mesh]
            Ui = self.lbs_matrix(xi, self.Q)
            self.U[i * nodes_per_mesh * 3: (i + 1) * nodes_per_mesh * 3, i * self.n_modes: (i + 1) * self.n_modes] = Ui

    def lbs_matrix(self, V, W):
        nvm = V.shape[0]
        v1 = np.ones((nvm, 4))
        v1[:, :3] = V
        lhs = np.hstack([W[:, j: j + 1] * v1 for j in range(W.shape[1])])
        return np.kron(lhs, np.identity(3))
        
    def solve(self):
        self.A_reduced = (self.U.T @ self.to_scipy_bsr() @ self.U)
        b = self.b.numpy().reshape(-1)
        self.b_reduced = self.U.T @ b
        dz = solve(self.A_reduced, self.b_reduced, assume_a = "sym")
        dx = (self.U @ dz).reshape(-1, 3) + self.comp_x.numpy()
        print(f"dx[1] = {np.max(np.abs(dx[:, 1]))}")
        self.states.dx.assign(dx)
        
        
@wp.kernel
def fill_J_triplets(xcs: wp.array(dtype = wp.vec3), W: wp.array2d(dtype = float), triplets: Triplets):
    i, j, k = wp.tid()
    xx = W.shape[0]
    yy = W.shape[1]

    idx = (i * yy + j) * 4 + k
    triplets.rows[idx] = i
    triplets.cols[idx] = j * 4 + k
    c = float(1.0)
    if k < 3:
        c = xcs[i][k]
    triplets.vals[idx] = wp.diag(wp.vec3(W[i, j] * c))


@wp.kernel
def update_z0_zdot(z: wp.array(dtype = wp.vec3), zdot: wp.array(dtype = wp.vec3), z0: wp.array(dtype = wp.vec3), dz: wp.array(dtype = wp.vec3), h: float):
    i = wp.tid()
    z[i] = z0[i] + dz[i]
    zdot[i] = dz[i] / h
    z0[i] = z[i]

def reduced_bunny_rain():
    n_meshes = 10
    meshes = ["assets/bunny_5.tobj"] * n_meshes
    
    transforms = wp.zeros((n_meshes, ), dtype = wp.mat44)
    v, _ = import_tobj(meshes[0])
    bb_size = np.max(v, axis = 0) - np.min(v, axis = 0)
    wp.launch(init_transforms, (n_meshes,), inputs = [transforms, bb_size[0], bb_size[1], bb_size[2]])
    print(f"bb_size = {bb_size}")
    rods = RodComplexBC(h, meshes, transforms.numpy())
    viewer = PSViewer(rods)
    ps.set_user_callback(viewer.callback)
    ps.show()

def spin():
    n_meshes = 1
    meshes = ["assets/bar2.tobj"] * n_meshes
    
    # transforms = wp.zeros((n_meshes, 4, 4), dtype = float)
    transforms = [np.identity(4, dtype = float) for _ in range(n_meshes)]
    transforms = np.array(transforms, dtype = float)
    transforms[0, :3, 3] = np.array([0.0, 2.0, 0.0])

    # rod = ReducedRod(h)
    rod = ReducedRodComplex(h, meshes, transforms)
    viewer = PSViewer(rod)
    ps.set_user_callback(viewer.callback)
    ps.show()

def staggered_bars():
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
    
    rods = ReducedRodComplex(h, meshes, transforms)
    viewer = PSViewer(rods)
    ps.set_user_callback(viewer.callback)
    ps.show()

def staggered_bug():
    
    n_meshes = 1
    # meshes = ["assets/bug.tobj"] * n_meshes
    meshes = ["assets/squishy/squishy.tobj"] * n_meshes
    # meshes = ["assets/bunny_5.tobj"] * n_meshes
    transforms = [np.identity(4, dtype = float) for _ in range(n_meshes)]
    # transforms[1][:3, :3] = np.zeros((3, 3))
    # transforms[1][0, 1] = 1
    # transforms[1][1, 0] = 1
    # transforms[1][2, 2] = 1

    for i in range(n_meshes):
        # transforms[i][0, 3] = i * 0.5
        transforms[i][1, 3] = 0.7 + i * 0.25
        transforms[i][2, 3] = i * 1.2 - 0.4
    
    # rods = MedialRodComplex(h, meshes, transforms)
    static_meshes_file = ["assets/teapotContainer.obj"]
    scale = np.identity(4) * 3
    scale[3, 3] = 1.0
    # static_bars = StaticScene(static_meshes_file, np.array([scale]))
    static_bars = None
    rods = ReducedRodComplex(h, meshes, transforms)
    
    
    viewer = PSViewer(rods, static_bars)
    ps.set_user_callback(viewer.callback)
    ps.show()

def twist():
    n_meshes = 1
    meshes = ["assets/bar2.tobj"]
    transfroms = [np.identity(4, dtype = float)]
    # transfroms[0][1, 3] = 0.2
    rods = ReducedRodComplex(h, meshes, transfroms)
    viewer = PSViewer(rods)
    
    ps.set_user_callback(viewer.callback)
    ps.show()

if __name__ == "__main__":  
    ps.init() 
    ps.set_ground_plane_height(-collision_eps)
    # ps.set_ground_plane_mode("none")
    wp.config.max_unroll = 0
    wp.init()
    
    # reduced_bunny_rain()
    # spin()
    # staggered_bars()
    staggered_bug()
    # twist()
    