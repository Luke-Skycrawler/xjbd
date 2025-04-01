import warp as wp
import numpy as np
from stretch import NewtonState, Triplets, h, add_dx, PSViewer
from mesh_complex import RodComplexBC, init_transforms, collision_eps
import polyscope as ps
import polyscope.imgui as gui
from tobj import import_tobj
from scipy.linalg import solve
from reduced import init_spin
from ortho import OrthogonalEnergy
from warp.sparse import bsr_axpy

class AffineBodySimulator(RodComplexBC):
    def __init__(self, h, meshes = [], transforms = []):
        
        super().__init__(h, meshes, transforms)

        self.define_U()
        self.A_reduced = np.zeros((self.n_reduced, self.n_reduced))
        self.b_reduced = np.zeros((self.n_reduced, ))
        self.ortho = OrthogonalEnergy()
        self.mm = self.U.T @ self.to_scipy_bsr(self.M_sparse) @ self.U


    def define_U(self):
        n_meshes = len(self.meshes_filename)
        nodes_per_model = self.n_nodes // n_meshes
        x0 = self.xcs.numpy()[:nodes_per_model]
        v1 = np.ones((x0.shape[0], 4))
        v1[:, :3] = x0
        self.U = np.kron(np.identity(n_meshes,), np.kron(v1, np.identity(3,)))
        self.n_reduced = 12 * n_meshes



    def add_collision_to_sys_matrix(self, triplets):
        super().add_collision_to_sys_matrix(triplets)
        self.A_reduced = self.U.T @ self.to_scipy_bsr() @ self.U

    def compute_rhs(self):
        super().compute_rhs()
        b = self.b.numpy().reshape(-1)
        self.b_reduced = self.U.T @ b
        

    
    def solve(self):
        dz = solve(self.A_reduced, self.b_reduced, assume_a = "sym")
        dx = (self.U @ dz).reshape(-1, 3) + self.comp_x.numpy()
        self.z -= dz.reshape(-1)
        # print(f"dz = {dz}, z = {self.z}")
        self.states.dx.assign(dx)
        
    
    def line_search(self):
        # wp.launch(add_dx, self.n_nodes, inputs = [self.states, 1.0])
        self.states.x.assign((self.U @ self.z).reshape((-1, 3)))
        return 1.0

class AffineBodySimulatorDebug(AffineBodySimulator):
    def __init__(self, h, meshes = [], transforms = []):
        super().__init__(h, meshes, transforms)

    def reset(self):
        self.z = np.zeros(12)
        for i in range(3):
            self.z[i * 4] = 1.0
        self.theta = 0.0
        n_verts = 525
        wp.copy(self.states.x, self.xcs)
        wp.copy(self.states.x0, self.xcs)

        # pos = self.transforms[:, :3, 3]
        # positions = wp.array(pos, dtype = wp.vec3)
        print("init spin")
        wp.launch(init_spin, (self.n_nodes,), inputs = [self.states])
        print("xdot = ", self.states.xdot.numpy())

    def add_collision_to_sys_matrix(self, triplets):
        self.A_reduced = self.U.T @ self.to_scipy_bsr() @ self.U

        sum_W = np.sum(self.W.numpy())
        F = np.zeros((3, 3))
        for i in range(3):
            F[:, i] = self.z[i * 3: i * 3 + 3]
        d2Psi = self.ortho.hessian(F)
        
        Aw = sum_W * d2Psi * self.h * self.h + self.mm
        # print(f"A reduced = {self.A_reduced[:3, :3]}, Aw = {Aw[:3, :3]}")
        # print(f"d2Psi = {Kw[:3, :3]}, Hw = {sum_W * d2Psi[:3, :3]}")
        self.A_reduced[:] = Aw


def bar_rain():
    n_meshes = 10
    meshes = ["assets/bar2.tobj"] * n_meshes
    
    transforms = wp.zeros((n_meshes, ), dtype = wp.mat44)
    v, _ = import_tobj(meshes[0])
    # bb_size = np.max(v, axis = 0) - np.min(v, axis = 0)
    bb_size = np.ones(3,)
    wp.launch(init_transforms, (n_meshes,), inputs = [transforms, bb_size[0], bb_size[1], bb_size[2]])
    print(f"bb_size = {bb_size}")
    rods = AffineBodySimulator(h, meshes, transforms.numpy())
    viewer = PSViewer(rods)
    ps.set_user_callback(viewer.callback)
    ps.show()

def spin():
    n_meshes = 1
    n_meshes = 1
    meshes = ["assets/bar2.tobj"] * n_meshes
    
    # transforms = wp.zeros((n_meshes, 4, 4), dtype = float)
    transforms = [np.identity(4, dtype = float) for _ in range(n_meshes)]
    transforms = np.array(transforms, dtype = float)
    transforms[0, :3, 3] = np.array([0.0, 2.0, 0.0])

    # rod = ReducedRod(h)
    rod = AffineBodySimulatorDebug(h, meshes, transforms)
    viewer = PSViewer(rod)
    ps.set_user_callback(viewer.callback)
    ps.show()

if __name__ == "__main__":
    ps.init()
    # ps.set_ground_plane_mode("none")
    ps.set_ground_plane_height(-collision_eps)
    wp.config.max_unroll = 0
    wp.init()
    spin()
    # bar_rain()