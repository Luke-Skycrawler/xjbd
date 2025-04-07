import warp as wp
from fem.interface import RodComplex
from fem.params import FEMMesh
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
from scipy.linalg import solve, cho_factor, cho_solve, polar
from scipy.io import loadmat
from igl import lbs_matrix
from fast_cd import CSRTriplets, compute_Hw
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from ortho import OrthogonalEnergy

spin_y = True
ref_A_reduced = False

@wp.kernel
def per_node_forces(geo: FEMMesh, b: wp.array(dtype = wp.vec3), h: float):
    i = wp.tid()
    xi = geo.xcs[i][0]
    fz = wp.cos(2. * wp.pi * xi) * 20.0
    eps = 1e-3
    if xi < -0.5 + eps or xi > 0.5 - eps:
        fz *= 0.5
    b[i] -= h * h * wp.vec3(0., 0., fz)
    
def orthogonalize(U, eps=1e-15):
    """
    Orthogonalizes the matrix U (d x n) using Gram-Schmidt Orthogonalization.
    If the columns of U are linearly dependent with rank(U) = r, the last n-r columns 
    will be 0.
    
    Args:
        U (numpy.array): A d x n matrix with columns that need to be orthogonalized.
        eps (float): Threshold value below which numbers are regarded as 0 (default=1e-15).
    
    Returns:
        (numpy.array): A d x n orthogonal matrix. If the input matrix U's cols were
            not linearly independent, then the last n-r cols are zeros.
    
    Examples:
    ```python
    >>> import numpy as np
    >>> import gram_schmidt as gs
    >>> gs.orthogonalize(np.array([[10., 3.], [7., 8.]]))
    array([[ 0.81923192, -0.57346234],
       [ 0.57346234,  0.81923192]])
    >>> gs.orthogonalize(np.array([[10., 3., 4., 8.], [7., 8., 6., 1.]]))
    array([[ 0.81923192 -0.57346234  0.          0.        ]
       [ 0.57346234  0.81923192  0.          0.        ]])
    ```
    """
    
    n = len(U[0])
    # numpy can readily reference rows using indices, but referencing full rows is a little
    # dirty. So, work with transpose(U)
    V = U.T
    for i in range(12, n):
        prev_basis = V[0:12]     # orthonormal basis before V[i]
        coeff_vec = np.dot(prev_basis, V[i].T)  # each entry is np.dot(V[j], V[i]) for all j < i
        # subtract projections of V[i] onto already determined basis V[0:i]
        V[i] -= np.dot(coeff_vec, prev_basis).T
        if np.linalg.norm(V[i]) < eps:
            V[i][V[i] < eps] = 0.   # set the small entries to 0
        else:
            V[i] /= np.linalg.norm(V[i])
            
    return V.T

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

def asym(a):
    return 0.5 * (a - a.T)

class ReducedRodComplex(RodComplexBC):
    def __init__(self, h, meshes = [], transforms = []) :
        self.z = np.zeros(120)
        self.z_dot = np.zeros_like(self.z)
        self.z0 = np.zeros_like(self.z)
        self.z_tilde = np.zeros_like(self.z[12:])
        self.z_tilde_dot = np.zeros_like(self.z_tilde)
        self.z_tilde0 = np.zeros_like(self.z_tilde)

        super().__init__(h, meshes, transforms)

        self.define_Hw()
        self.define_U()
        n_reduced = self.n_reduced
        self.A_reduced = np.zeros((n_reduced, n_reduced))
        self.b_reduced = np.zeros(n_reduced)
        self.ortho = OrthogonalEnergy()
        self.sum_W = np.sum(self.W.numpy())
        self.mm = self.U0.T @ self.to_scipy_bsr(self.M_sparse) @ self.U0
        self.prefactor_once()
    
    def get_F(self):
        '''
        columns of F is formed by segments of z 
        vec(F) = z
        '''
        F = self.z.reshape((-1, 3)).T
        return F[:, :3]

    def eigs(self):
        # FIXME: don't call this if there is more than 1 object
        assert(len(self.meshes_filename) == 1)

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

        n_meshes = len(self.meshes_filename)
        nodes_per_model = self.n_nodes // n_meshes
        x0 = self.xcs.numpy()[:nodes_per_model]
        v1 = np.ones((x0.shape[0], 4))
        v1[:, :3] = x0
        uu = np.kron(np.identity(n_meshes,), np.kron(v1, np.identity(3,)))

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
            self.U[:, : 12] = uu
        else:
            n, m = U.shape
            self.U = np.zeros((self.n_nodes * 3, self.n_modes * 12 * n_objects))
            for i in range(n_objects):
                self.U[i * n: (i + 1) * n, i * m: (i + 1) * m] = np.copy(U)
        
        self.U = orthogonalize(U)
        self.U0 = uu
        self.U_tilde = self.U[:, 12:]
        self.n_reduced = n_objects * self.n_modes * 12
        UTU = self.U_tilde.T @ self.U_tilde
        Lam = np.diag(UTU)
        diag_lam = np.zeros(self.n_nodes * 3, )
        diag_lam[:Lam.shape[0]] = 1.0 / Lam
        self.diag_lam = diag_lam
        # self.UT_tilde = self.U_tilde.T @ np.diag(diag_lam)

        # self.UT_tilde = np.linalg.inv(UTU) @ self.U_tilde.T
        self.UT_tilde = self.U_tilde.T

    def add_collision_to_sys_matrix(self, triplets):
        # super().add_collision_to_sys_matrix(triplets)
        self.A_reduced = (self.U.T @ self.to_scipy_bsr() @ self.U)        
        pass
        
    def compute_rhs(self):
        super().compute_rhs()
        # if np.cos(self.theta * 2) > 0.0:
        #     wp.launch(per_node_forces, (self.n_nodes, ), inputs = [self.geo, self.b, self.h])
        b = self.b.numpy().reshape(-1) + self.per_node_forces()
        self.b_reduced = self.U.T @ b
    
    def tilde_z(self):
        return self.z - (self.h * self.z_dot + self.z0)

    def tilde_z0(self):
        return self.z[:12] - (self.h * self.z_dot[:12] + self.z0[:12])

    def tilde_z_tilde(self):
        return self.h * self.z_tilde_dot + self.z_tilde0

    def update_x0_xdot(self):
        super().update_x0_xdot()
        self.z_dot[:] = (self.z - self.z0) / self.h
        self.z0[:] = self.z[:]

        self.z_tilde_dot[:] = (self.z_tilde - self.z_tilde0) / self.h
        self.z_tilde0[:] = self.z_tilde

    def prefactor_once(self):
        h = self.h
        self.compute_K()
        self.K0 = self.U_tilde.T @ self.to_scipy_bsr() * h * h @  self.U_tilde
        self.M_tilde = self.U_tilde.T @ self.to_scipy_bsr(self.M_sparse) @ self.U_tilde
        # self.K0 = self.U.T @ self.to_scipy_bsr() * h * h @  self.U
        # self.M_tilde = self.U.T @ self.to_scipy_bsr(self.M_sparse) @ self.U
        self.A_tilde = self.K0 + self.M_tilde
        self.c, self.low = cho_factor(self.A_tilde)

    def compute_A(self):
        if ref_A_reduced:
            super().compute_A()
        else:
            pass
    
    def get_R(self):
        F = self.get_F()
        return F
        # u, p = polar(F)
        return u

    def dz_tilde2dz(self, dz_tilde, dz0):
        dz = np.zeros_like(dz_tilde)
        R = self.get_R()

        dzt = dz_tilde.reshape((-1, 3)).T
        dR = dz0.reshape((-1, 3)).T[:, :3]
        # dR = asym(dR)
        zt = (self.z_tilde.reshape((-1, 3))).T
        dz = (R @ dzt + dR @ zt).T.reshape(-1)
        # dz = (R @ dzt).T.reshape(-1)
        return dz

    def solve(self):
        F = self.get_F()
        g, d2Psi = self.ortho.analyze(F)
        self.A0 = self.sum_W * d2Psi * self.h * self.h + self.mm
        bw = self.sum_W * g * self.h * self.h + self.mm @ self.tilde_z0()
        self.b0 = bw

        # self.b0 = self.U0.T @ self.b.numpy().reshape(-1)
        # self.A0 = self.U0.T @ self.to_scipy_bsr() @ self.U0

        dz0 = solve(self.A0, self.b0, assume_a = "sym")
        self.b_tilde = self.K0 @ self.z_tilde + self.M_tilde @ (self.z_tilde - self.tilde_z_tilde()) + self.compute_excitement()

        dz_tilde = cho_solve((self.c, self.low), self.b_tilde)
        # b1 = self.M_tilde @ self.tilde_z_tilde()
        # z1 = cho_solve((self.c, self.low, ), b1)

        # dz_tilde = self.z_tilde - z1
        # self.z_tilde = z1

        self.z_tilde -= dz_tilde

        dz = np.zeros_like(self.z)
        dz[:12] = dz0[:12]
        dz[12:] = self.dz_tilde2dz(dz_tilde, dz0)

        if ref_A_reduced:
            dz = solve(self.A_reduced, self.b_reduced, assume_a = "sym")

        # if hasattr(self, "cnt"):
        #     self.cnt += 1
        # else:
        #     self.cnt = 0
        # np.save(f"output/A_reduced{self.cnt}.npy", self.A_reduced)
        # np.save(f"output/b_reduced{self.cnt}.npy", self.b_reduced)
        # np.save(f"output/b_tilde{self.cnt}.npy", self.b_tilde)
        # np.save(f"output/A_tilde.npy", self.A_tilde)
        self.z -= dz
        
        dx = (self.U @ dz).reshape(-1, 3) + self.comp_x.numpy()
        # print(f"dx[1] = {np.max(np.abs(dx[:, 1]))}")
        self.states.dx.assign(dx)
        
    
    def line_search(self):
        # wp.launch(add_dx, self.n_nodes, inputs = [self.states, 1.0])
        self.states.x.assign((self.U @ self.z).reshape((-1, 3)))
        return 1.0

    def reset(self):
        self.theta = 0.0
        n_verts = 525
        wp.copy(self.states.x, self.xcs)
        wp.copy(self.states.x0, self.xcs)

        # pos = self.transforms[:, :3, 3]
        # positions = wp.array(pos, dtype = wp.vec3)
        if spin_y:
            wp.launch(init_spin, (self.n_nodes,), inputs = [self.states])
            print("xdot = ", self.states.xdot.numpy())
        self.reset_z()

    def reset_z(self):
        n_meshes = len(self.meshes_filename)
        n_modes = len(self.z) // n_meshes
        for j in range(n_meshes):
            for i in range(3):
                self.z[12 * j * n_modes + i * 4] = 1.0

        self.z0[:] = self.z[:]
        self.z_dot[:] = 0.0

        if spin_y:
            self.z_dot[2] = -1.0
            self.z_dot[6] = 1.0

        self.z_tilde[:] = 0.0
        self.z_tilde0[:] = 0.0
        self.z_tilde_dot[:] = 0.0

    def per_node_forces(self):
        b = wp.zeros_like(self.b)
        if np.cos(self.theta * 2) > 0.0:
        # if False:
            wp.launch(per_node_forces, (self.n_nodes, ), inputs = [self.geo, b, self.h])
        return b.numpy().reshape(-1)

    def compute_excitement(self):
        # FIXME: check sign here
        f = self.per_node_forces()# / self.sum_W
        R = self.get_R()
        # i_m = np.zeros(((self.n_modes - 1) * 4, (self.n_modes - 1) * 4), float)
        i_m = np.identity((self.n_modes - 1) * 4, float)
        
        # Q_tilde = (self.U_tilde @ np.kron(i_m, R.T)).T @ np.diag(self.diag_lam) @ f
        # Q_tilde = (self.U_tilde @ np.kron(i_m, R.T)).T @ f
        Q_tilde = np.kron(i_m, R) @ self.UT_tilde @ f
        return Q_tilde

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
    transforms[0, :3, 3] = np.array([0.0, 0.0, 0.0])
    # don't change it back to [0, 2, 0], cho factor will fail

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
    transforms[1][:3, :3] = np.zeros((3, 3))
    transforms[1][0, 1] = 1
    transforms[1][1, 0] = 1
    transforms[1][2, 2] = 1
    
    for i in range(n_meshes):
        # transforms[i][0, 3] = i * 0.5
        transforms[i][1, 3] = 1.2
        transforms[i][2, 3] = i * 1.0
    
    rods = ReducedRodComplex(h, meshes, transforms)
    viewer = PSViewer(rods)
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
    ps.look_at((0.0, 4.0, 0.1), (0.0, -2.0, 0.0))
    # ps.set_ground_plane_height(-collision_eps)
    ps.set_ground_plane_mode("none")
    wp.config.max_unroll = 0
    wp.init()
    
    # reduced_bunny_rain()
    spin()
    # staggered_bars()
    # twist()
    