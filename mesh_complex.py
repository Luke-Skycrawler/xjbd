import warp as wp
from stretch import RodBCBase, PSViewer, NewtonState, Triplets, add_dx
from fem.interface import Rod, RodComplex
import polyscope as ps
import numpy as np
# collision add-ons
from geometry.collision_cell import MeshCollisionDetector

from warp.sparse import bsr_axpy, bsr_set_from_triplets, bsr_zeros
h = 1e-2
        
class RodComplexBC(RodBCBase, RodComplex):
    def __init__(self, h, meshes = [], transforms = []):
        self.meshes_filename = meshes 
        self.transforms = transforms
        super().__init__(h)
        self.collider = MeshCollisionDetector(self.states.x, self.T, self.indices, self.Bm, ground = 0.0)
        self.n_pt = 0
        self.n_ee = 0
        self.n_ground = 0

    def reset(self):
        n_verts = 4
        if self.meshes_filename[0] == "assets/bar2.tobj":
            n_verts = 525
        elif self.meshes_filename[0] == "assets/tet.tobj":
            n_verts = 4
        elif self.meshes_filename[0] == "assets/bunny_5.tobj":
            n_verts = 1356
        wp.copy(self.states.x, self.xcs)
        wp.copy(self.states.x0, self.xcs)
        if self.meshes_filename[0] == "assets/tet.tobj":
            wp.launch(set_vx_kernel, (self.n_nodes,), inputs = [self.states, n_verts])
        else: 
            wp.launch(set_velocity_kernel, (self.n_nodes,), inputs = [self.states, n_verts])
        

    def set_bc_fixed_hessian(self):
        pass

    def set_bc_fixed_grad(self):
        pass

    def step(self):
        newton_iter = True
        n_iter = 0
        max_iter = 5
        # while n_iter < max_iter:
        while newton_iter:
            self.compute_A()
            self.n_pt, self.n_ee, self.n_ground = self.collider.collision_set("all") 
            triplets = self.collider.analyze(self.b, self.n_pt, self.n_ee, self.n_ground)
            self.compute_rhs()
            self.add_collision_to_sys_matrix(triplets)

            self.solve()
            # wp.launch(add_dx, dim = (self.n_nodes, ), inputs = [self.states, 1.0])
            
            dxnp = self.states.dx.numpy()
            norm_dx = np.linalg.norm(dxnp)
            newton_iter = norm_dx > 1e-3 and n_iter < max_iter
            if norm_dx < 1e-5:
                break

            # line search stuff, not converged yet
            alpha = self.line_search()
            if alpha == 0.0:
                break

            print(f"norm = {np.linalg.norm(dxnp)}, {n_iter}")
            n_iter += 1
        self.update_x0_xdot()

    def add_collision_to_sys_matrix(self, triplets: Triplets):

        collision_force_derivatives = bsr_zeros(self.n_nodes, self.n_nodes, wp.mat33)
        bsr_set_from_triplets(collision_force_derivatives, triplets.rows, triplets.cols, triplets.vals)
        bsr_axpy(collision_force_derivatives, self.A, self.h * self.h)

    def compute_collision_energy(self):
        return self.collider.collision_energy(self.n_pt, self.n_ee, self.n_ground) * self.h * self.h
        # return 0.0

def multiple_drape():
    n_meshes = 3
    meshes = ["assets/bar2.tobj"] * n_meshes
    transforms = [np.identity(4, dtype = float) for _ in range(n_meshes)]
    for i in range(n_meshes):
        transforms[i][2, 3] = i * 2.0

    rods = RodComplexBC(h, meshes, transforms)
    viewer = PSViewer(rods)
    ps.set_user_callback(viewer.callback)
    ps.show()


@wp.kernel
def set_velocity_kernel(states: NewtonState, thres: int):
    i = wp.tid()
    states.xdot[i] = wp.vec3(0.0)
    if i >= thres:
        states.xdot[i] = wp.vec3(0.0, 0.0, -3.0)

@wp.kernel
def set_vx_kernel(states: NewtonState, thres: int):
    i = wp.tid()
    states.xdot[i] = wp.vec3(0.0)
    if i >= thres:
        states.xdot[i] = wp.vec3(3.0, 0.0, 0.0)

def staggered_bars():
    n_meshes = 2 
    meshes = ["assets/bar2.tobj"] * n_meshes
    # meshes = ["assets/bunny_5.tobj"] * n_meshes
    transforms = [np.identity(4, dtype = float) for _ in range(n_meshes)]
    for i in range(n_meshes):
        transforms[i][0, 3] = i * 0.5
        transforms[i][1, 3] = 0.2
        transforms[i][2, 3] = i * 1.0
    
    rods = RodComplexBC(h, meshes, transforms)
    viewer = PSViewer(rods)
    ps.set_user_callback(viewer.callback)
    ps.show()

def tets():
    n_meshes = 2 
    meshes = ["assets/tet.tobj"] * n_meshes
    transforms = [np.identity(4, dtype = float) for _ in range(n_meshes)]
    for i in range(n_meshes):
        transforms[i][0, 3] = i * -2.0
        transforms[i][1, 3] = i * 0.5
        transforms[i][2, 3] = i * 0.5
    
    rods = RodComplexBC(h, meshes, transforms)
    viewer = PSViewer(rods)
    ps.set_user_callback(viewer.callback)
    ps.show()

if __name__ == "__main__":
    ps.init()
    # ps.set_ground_plane_mode("none")
    ps.set_ground_plane_height(0.0)
    wp.config.max_unroll = 0
    wp.init()

    # multiple_drape()
    # drape()
    staggered_bars()
    # tets()