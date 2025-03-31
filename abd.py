import warp as wp
import numpy as np
from stretch import NewtonState, Triplets, h, add_dx, PSViewer
from mesh_complex import RodComplexBC, init_transforms
import polyscope as ps
import polyscope.imgui as gui

from scipy.linalg import solve
from reduced import init_spin

class AffineBodySimulator(RodComplexBC):
    def __init__(self, h, meshes = [], transforms = []):
        super().__init__(h, meshes, transforms)

        self.define_U()
        self.A_reduced = np.zeros((self.n_reduced, self.n_reduced))
        self.b_reduced = np.zeros((self.n_reduced, ))

    def reset(self):
        self.theta = 0.0
        n_verts = 525
        wp.copy(self.states.x, self.xcs)
        wp.copy(self.states.x0, self.xcs)

        # pos = self.transforms[:, :3, 3]
        # positions = wp.array(pos, dtype = wp.vec3)
        print("init spin")
        wp.launch(init_spin, (self.n_nodes,), inputs = [self.states])
        print("xdot = ", self.states.xdot.numpy())

    def define_U(self):

        x0 = self.xcs.numpy()
        v1 = np.ones((x0.shape[0], 4))
        v1[:, :3] = x0
        self.U = np.kron(x0, np.identity(3,))
        self.n_reduced = 12



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
        print(f"dx[1] = {np.max(np.abs(dx[:, 1]))}")
        self.states.dx.assign(dx)
        
    
    def line_search(self):
        wp.launch(add_dx, self.n_nodes, inputs = [self.states, 1.0])
        return 1.0


def spin():
    n_meshes = 1
    n_meshes = 1
    meshes = ["assets/bar2.tobj"] * n_meshes
    
    # transforms = wp.zeros((n_meshes, 4, 4), dtype = float)
    transforms = [np.identity(4, dtype = float) for _ in range(n_meshes)]
    transforms = np.array(transforms, dtype = float)
    transforms[0, :3, 3] = np.array([0.0, 2.0, 0.0])

    # rod = ReducedRod(h)
    rod = AffineBodySimulator(h, meshes, transforms)
    viewer = PSViewer(rod)
    ps.set_user_callback(viewer.callback)
    ps.show()

if __name__ == "__main__":
    ps.init()
    ps.set_ground_plane_mode("none")
    wp.config.max_unroll = 1
    wp.init()
    spin()