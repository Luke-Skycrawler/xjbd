import warp as wp
from fem.interface import RodComplex
from stretch import RodBCBase, NewtonState, Triplets, h
from mesh_complex import RodComplexBC, init_transforms

import polyscope as ps
import polyscope.imgui as gui
import numpy as np

from geometry.collision_cell import MeshCollisionDetector, collision_eps
from utils.tobj import import_tobj
from warp.sparse import bsr_axpy, bsr_set_from_triplets, bsr_zeros, bsr_mm, bsr_transposed, bsr_mv
from fast_cd import RodLBSWeight
import os 
from scipy.linalg import cholesky, cho_solve, cho_factor
from igl import lbs_matrix
class ReducedRodComplex(RodComplexBC):
    def __init__(self, h, meshes = [], transforms = []):
        super().__init__(h, meshes, transforms)


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

class ReducedRod(RodLBSWeight):
    def __init__(self, h):
        super().__init__()
        self.h = h
        self.define_UTKU()


        # self.z = wp.zeros((self.n_modes * 4,), dtype = wp.vec3)
        # self.zdot = wp.zeros_like(self.z)
        # self.z0 = wp.zeros_like(self.z)
        # self.dz = wp.zeros_like(self.z)
        # z - tilde z
        self.z = np.zeros((self.n_modes * 12,), dtype = float)
        self.zdot = np.zeros_like(self.z)
        self.z0 = np.zeros_like(self.z)
        self.dz = np.zeros_like(self.z)

        for i in range(10):
            self.z0[i * 12: i * 12 + 12] = np.eye(4, 3, dtype = float).reshape(-1)
            self.z[i * 12: i * 12 + 12] = np.eye(4, 3, dtype = float).reshape(-1)
    def define_UTKU(self):
        model = "bar2"
        Q = None
        if not os.path.exists(f"data/W_{model}.npy"):
        # if True:
            _, Q = self.eigs()
            np.save(f"data/W_{model}.npy", Q)
        else:
            Q = np.load(f"data/W_{model}.npy")

        self.weights = wp.array(Q, dtype = float)
        
        self.n_modes = Q.shape[1]
        
        self.J_triplets = Triplets()
        self.J_triplets.rows = wp.zeros((self.n_modes * self.n_nodes * 4,), dtype = int)
        self.J_triplets.cols = wp.zeros_like(self.J_triplets.rows)  
        self.J_triplets.vals = wp.zeros((self.n_modes * self.n_nodes * 4,), dtype = wp.mat33)

        wp.launch(fill_J_triplets, (self.weights.shape[0], self.weights.shape[1], 4), inputs = [self.xcs, self.weights, self.J_triplets])

        # asssemble [0, C^T; C, 0]
        self.U = bsr_zeros(self.n_nodes, self.n_modes * 4, wp.mat33)        
        
        bsr_set_from_triplets(self.U, self.J_triplets.rows, self.J_triplets.cols, self.J_triplets.vals)

        self.uu = self.to_scipy_bsr(self.U)
        self.kk = self.to_scipy_bsr(self.K_sparse)

        self.uktu = (self.uu.transpose() @ self.kk @ self.uu).toarray()
        self.sys_matrix = self.h * self.h * self.uktu + np.identity(self.n_modes * 12)

        self.B = lbs_matrix(self.xcs.numpy(), Q)
        self.T = np.zeros((4 * self.n_modes, 3), dtype = float)
        self.T[:3, :] = np.eye(3)
        
        self.verify_J()
        

        
        # self.UTKU = bsr_mm(bsr_transposed(self.U), bsr_mm(self.K_sparse, self.U), alpha = self.h * self.h)
        
        # self.sys_matrix = self.to_scipy_bsr(self.UTKU).toarray() + np.identity(self.n_modes * 12)

        self.c, self.lower = cho_factor(self.sys_matrix, lower = True)
    
    def verify_J(self):
        i = np.random.randint(0, self.n_nodes)
        # j = np.random.randint(0, self.n_modes * 12)  
        j = np.random.randint(0, 2 * 12)  
        Tp = np.copy(self.T)
        Tn = np.copy(self.T)
        dt = 1e-2
        Tp[j // 3, j % 3] = +dt
        Tn[j // 3, j % 3] = -dt
        dx = self.B @ (Tp - Tn)
        dx_ana = self.uu @ (Tp - Tn).reshape(-1)

        print(f"i = {i}, j = {j}, dx = {dx}, dx_ana = {dx_ana}")
        print(f"diff = {np.linalg.norm(dx.reshape(-1) - dx_ana)}")
        print(f"dx norm = {np.linalg.norm(dx.reshape(-1))}")
        quit()
    def step(self):
        b = self.compute_rhs()
        dx = self.solve(b)
        self.update_x0_xdot(dx)
        

    def solve(self, b):
        dx = cho_solve((self.c, self.lower), b)
        return dx

    def update_x0_xdot(self, dx):
        # self.dz.assign(dx.reshape(-1, 3))
        # wp.launch(update_z0_zdot, (self.n_modes * 4,), inputs = [self.z, self.zdot, self.z0, self.dz, self.h])
        self.zdot = -dx / self.h
        self.z = self.z0 - dx
        self.z0 = self.z
        
        
    def compute_rhs(self):
        # wp.launch(inertia_term, (self.n_modes * 4), input = [self.z, self.zdot, self.z0, self.dz, self.h])
        self.dz = self.z - (self.z0 + self.h * self.zdot)
        b = self.uktu @ self.z * self.h * self.h + self.dz
        return b
        # bsr_mv(self.UTKU, self.z, self.dz, alpha = self.h * self.h, beta  = 1.0)
        # return self.dz.numpy().reshape(-1)
        
class PSViewer:
    def __init__(self, rod: ReducedRod):
        self.V0 = rod.xcs.numpy()
        self.F = rod.F

        self.ps_mesh = ps.register_surface_mesh("rod", self.V0, self.F)
        self.frame = 0
        self.rod = rod
        self.ui_pause = True
        self.animate = False
    def callback(self):
        changed, self.ui_pause = gui.Checkbox("Pause", self.ui_pause)
        self.animate = gui.Button("Step") or not self.ui_pause
        if gui.Button("Reset"):
            self.rod.reset()
            self.frame = 0
            self.ui_pause = True
            self.animate = True

        if self.animate: 
            self.rod.step()
            # self.V = self.rod.states.x.numpy()
            self.V = (self.rod.uu @ self.rod.z).reshape(-1, 3) - self.V0
            # print(f"V = {self.V}")
            self.ps_mesh.update_vertex_positions(self.V)
            self.frame += 1
            
            print("frame = ", self.frame)

            ps.screenshot(f"output/{self.frame:04d}.jpg")

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
    n_meshes = 10
    meshes = ["assets/bar2.tobj"] * n_meshes
    
    transforms = wp.zeros((n_meshes, 4, 4), dtype = float)
    

    rod = ReducedRod(h)
    viewer = PSViewer(rod)
    ps.set_user_callback(viewer.callback)
    ps.show()

if __name__ == "__main__":  
    ps.init() 
    ps.set_ground_plane_height(-collision_eps)
    wp.config.max_unroll = 0
    wp.init()
    
    # reduced_bunny_rain()
    spin()