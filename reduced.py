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

        self.define_Hw()
        self.define_U()
        n_reduced = self.n_reduced
        self.A_reduced = np.zeros((n_reduced, n_reduced))
        self.b_reduced = np.zeros(n_reduced)

    def eigs(self):
        K = self.to_scipy_csr()
        # print("start weight space eigs")
        with wp.ScopedTimer("weight space eigs"):
            lam, Q = eigsh(K, k = 10, which = "SM", tol = 1e-4)
            # Q_norm = np.linalg.norm(Q, axis = 0, ord = np.inf, keepdims = True)
            # Q /= Q_norm
        return lam, Q
        
    def to_scipy_csr(self):
        ii = self.Hw.offsets.numpy()
        jj = self.Hw.columns.numpy()
        values = self.Hw.values.numpy()

        csr = csr_matrix((values, jj, ii), shape = (self.n_nodes, self.n_nodes))
        return csr
        
    def define_Hw(self):
        self.triplets_Hw = CSRTriplets()
        self.triplets_Hw.rows = wp.zeros((self.n_tets * 4 * 4,), dtype = int)
        self.triplets_Hw.cols = wp.zeros_like(self.triplets_Hw.rows)
        self.triplets_Hw.vals = wp.zeros((self.n_tets * 4 *4), dtype = float)
        
        self.Hw = bsr_zeros(self.n_nodes, self.n_nodes, float)
        wp.launch(compute_Hw, (self.n_tets * 4 * 4,), inputs = [self.triplets, self.triplets_Hw])
        bsr_set_from_triplets(self.Hw, self.triplets_Hw.rows, self.triplets_Hw.cols, self.triplets_Hw.vals)


    def define_U(self):
        # self.n_modes = 1
        # self.U = np.kron(np.hstack((self.xcs.numpy(), np.ones((self.n_nodes, 1)))), np.identity(3, dtype = float))

        # self.U = dxdq_jacobian(self.n_nodes * 3, self.xcs.numpy())

        model = "bar2"
        Q = None
        if not os.path.exists(f"data/W_{model}.npy"):
        # if True:
            _, Q = self.eigs()
            np.save(f"data/W_{model}.npy", Q)
        else:
            Q = np.load(f"data/W_{model}.npy")

        print(f"Q = {Q.shape}, Q.variance = {np.var(Q)}, Q.mean = {np.mean(Q)}")
        self.weights = wp.array(Q, dtype = float)
        
        n_objects = len(self.transforms)
        self.n_modes = Q.shape[1] # fixme: ad-hoc
        
        self.J_triplets = Triplets()
        self.J_triplets.rows = wp.zeros((self.n_modes * self.n_nodes * 4,), dtype = int)
        self.J_triplets.cols = wp.zeros_like(self.J_triplets.rows)  
        self.J_triplets.vals = wp.zeros((self.n_modes * self.n_nodes * 4,), dtype = wp.mat33)

        wp.launch(fill_J_triplets, (self.weights.shape[0], self.weights.shape[1], 4), inputs = [self.xcs, self.weights, self.J_triplets])

        # asssemble [0, C^T; C, 0]
        self.uu = bsr_zeros(self.n_nodes // n_objects, self.n_modes * 4, wp.mat33)        
        
        bsr_set_from_triplets(self.uu, self.J_triplets.rows, self.J_triplets.cols, self.J_triplets.vals)

        U = self.to_scipy_bsr(self.uu).toarray()
        if n_objects == 1:
            self.U = U
        else:
            n, m = U.shape
            self.U = np.zeros((self.n_nodes * 3, self.n_modes * 12 * n_objects))
            for i in range(n_objects):
                self.U[i * n: (i + 1) * n, i * m: (i + 1) * m] = np.copy(U)
        self.n_reduced = n_objects * self.n_modes * 12

    def add_collision_to_sys_matrix(self, triplets):
        super().add_collision_to_sys_matrix(triplets)
        self.A_reduced = (self.U.T @ self.to_scipy_bsr() @ self.U)
        
    def compute_rhs(self):
        super().compute_rhs()
        b = self.b.numpy().reshape(-1)
        self.b_reduced = self.U.T @ b
    
    def solve(self):
        dz = solve(self.A_reduced, self.b_reduced, assume_a = "sym")
        dx = (self.U @ dz).reshape(-1, 3)
        print(f"dx[1] = {np.max(np.abs(dx[:, 1]))}")
        self.states.dx.assign(dx)
        
    
    def line_search(self):
        wp.launch(add_dx, self.n_nodes, inputs = [self.states, 1.0])
        return 1.0

    # def reset(self):
    #     n_verts = 525
    #     wp.copy(self.states.x, self.xcs)
    #     wp.copy(self.states.x0, self.xcs)

    #     # pos = self.transforms[:, :3, 3]
    #     # positions = wp.array(pos, dtype = wp.vec3)
    #     print("init spin")
    #     wp.launch(init_spin, (self.n_nodes,), inputs = [self.states])
    #     print("xdot = ", self.states.xdot.numpy())
        
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
def inertia_term(z: wp.array(dtype = wp.vec3), zdot: wp.array(dtype = wp.vec3), z0: wp.array(dtype = wp.vec3), dz: wp.array(dtype = wp.vec3), h: float):
    i = wp.tid()
    dz[i] = z[i] - (z0[i] + h * zdot[i])

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
    meshes = ["assets/bar2.tobj"] * n_meshes
    # meshes = ["assets/bunny_5.tobj"] * n_meshes
    transforms = [np.identity(4, dtype = float) for _ in range(n_meshes)]
    for i in range(n_meshes):
        transforms[i][0, 3] = i * 0.5
        transforms[i][1, 3] = 0.2
        transforms[i][2, 3] = i * 1.0
    
    rods = ReducedRodComplex(h, meshes, transforms)
    viewer = PSViewer(rods)
    ps.set_user_callback(viewer.callback)
    ps.show()

if __name__ == "__main__":  
    ps.init() 
    ps.set_ground_plane_height(-collision_eps)
    wp.config.max_unroll = 0
    wp.init()
    
    # reduced_bunny_rain()
    # spin()
    staggered_bars()
    