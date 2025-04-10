import warp as wp
from stretch import RodBCBase, PSViewer, NewtonState, Triplets, add_dx, h
from fem.interface import Rod, RodComplex
import polyscope as ps
import numpy as np
# collision add-ons
from geometry.collision_cell import MeshCollisionDetector, collision_eps
from utils.tobj import import_tobj
from warp.sparse import bsr_axpy, bsr_set_from_triplets, bsr_zeros

omega = 3.0

@wp.kernel
def init_velocities(states: NewtonState, positions: wp.array(dtype = wp.vec3), n_verts: int):
    i = wp.tid()
    j = i // n_verts
    pi = positions[j]
    v = wp.vec3(-pi[0], 0.0, -pi[2])
    states.xdot[i] = v * 0.5

class RodComplexBC(RodBCBase, RodComplex):
    def __init__(self, h, meshes = [], transforms = []):
        self.meshes_filename = meshes 
        self.transforms = transforms
        super().__init__(h)
        self.define_collider()
        self.n_pt = 0
        self.n_ee = 0
        self.n_ground = 0

    def define_collider(self):
        self.collider = MeshCollisionDetector(self.states.x, self.T, self.indices, self.Bm, ground = 0.0)

    def reset(self):
        n_verts = 4
        self.theta = 0.0
        if self.meshes_filename[0] == "assets/bar2.tobj":
            n_verts = 525
        elif self.meshes_filename[0] == "assets/tet.tobj":
            n_verts = 4
        elif self.meshes_filename[0] == "assets/bunny_5.tobj":
            n_verts = 1356
        elif self.meshes_filename[0] == "assets/bug.tobj":
            n_verts = 2471
        wp.copy(self.states.x, self.xcs)
        wp.copy(self.states.x0, self.xcs)
        if self.meshes_filename[0] == "assets/tet.tobj" or self.meshes_filename[1] == "assets/tet.tobj":
            wp.launch(set_vx_kernel, (self.n_nodes,), inputs = [self.states, n_verts])
        elif self.meshes_filename[0] == "assets/bar2.tobj" or self.meshes_filename[0] == "assets/bug.tobj": 
            wp.launch(set_velocity_kernel, (self.n_nodes,), inputs = [self.states, n_verts])
        else: 
            pos = self.transforms[:, :3, 3]
            positions = wp.array(pos, dtype = wp.vec3)
            wp.launch(init_velocities, (self.n_nodes,), inputs = [self.states, positions, n_verts])
        

    def set_bc_fixed_hessian(self):
        pass

    def set_bc_fixed_grad(self):
        pass
    
    def process_collision(self):
        with wp.ScopedTimer("collision"):
            with wp.ScopedTimer("detection"):
                self.n_pt, self.n_ee, self.n_ground = self.collider.collision_set("all") 
            with wp.ScopedTimer("hess & grad"):
                triplets = self.collider.analyze(self.b, self.n_pt, self.n_ee, self.n_ground)
                # triplets = self.collider.analyze(self.b)
            with wp.ScopedTimer("build_from_triplets"):
                self.add_collision_to_sys_matrix(triplets)
    def step(self):
        self.theta += omega * self.h
        with wp.ScopedTimer("step"):
            newton_iter = True
            n_iter = 0
            max_iter = 5
            # while n_iter < max_iter:
            while newton_iter:
                with wp.ScopedTimer(f"newton #{n_iter}"):
                    with wp.ScopedTimer("compute A"):
                        self.compute_A()
                    with wp.ScopedTimer("collision"):
                        self.process_collision()
                    self.compute_rhs()

                    with wp.ScopedTimer("solve"):
                        self.solve()
                    # wp.launch(add_dx, dim = (self.n_nodes, ), inputs = [self.states, 1.0])
                    
                    dxnp = self.states.dx.numpy()
                    norm_dx = np.linalg.norm(dxnp)
                    newton_iter = norm_dx > 1e-3 and n_iter < max_iter
                    if norm_dx < 1e-5:
                        break

                    # line search stuff, not converged yet
                    with wp.ScopedTimer("line search"):
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

@wp.func
def int_pow(base: int, exp: int, mod: int) -> int:
    result = int(1)
    x = int(base)
    n = int(exp)

    while n > 0:
        if n & 1:  # If n is odd, multiply by current base
            result *= x
            result %= mod
        x *= x  # Square the base
        x %= mod
        n >>= 1  # Divide exponent by 2

    return result

@wp.kernel(enable_backward=False)
def init_transforms(transforms: wp.array(dtype = wp.mat44), bb_x: float, bb_y: float, bb_z: float):
    i = wp.tid()
    bb = wp.vec3(bb_x, bb_y, bb_z)
    max_axis = wp.max(bb)
    scale = 1.0 / max_axis
    # ti = wp.diag(wp.vec4(scale, scale, scale, 1.0))
    ti = wp.mat44(0.0)
    ti[3, 3] = 1.0
    grid_dims = wp.vec3i(11, 13, 19)
    
    pos = wp.vec3(0.0)
    for dim in range(3):
        xd = int_pow(7, i, grid_dims[dim])
        pos[dim] = float(xd) / float(grid_dims[dim]) * 10.0 - 5.0
    pos[1] = wp.abs(pos[1]) + 1.0

    rng = wp.rand_init(123 + i)
    axis = wp.vec3(wp.randf(rng), wp.randf(rng), wp.randf(rng))
    axis = wp.normalize(axis)
    angle = wp.randf(rng) * 2.0 * 3.14159
    q = wp.quat_from_axis_angle(axis, angle)
    rot = wp.quat_to_matrix(q)
    
    # transforms[i] = rot @ ti    
    for dim in range(3):
        for jj in range(3):
            ti[dim, jj] = rot[dim, jj] * scale
        ti[dim, 3] = pos[dim]

    transforms[i] = ti


    
def bunny_rain():
    n_meshes = 20
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

def bar_rain():
    n_meshes = 10
    meshes = ["assets/bar2.tobj"] * n_meshes
    
    transforms = wp.zeros((n_meshes, ), dtype = wp.mat44)
    v, _ = import_tobj(meshes[0])
    bb_size = np.max(v, axis = 0) - np.min(v, axis = 0)
    wp.launch(init_transforms, (n_meshes,), inputs = [transforms, bb_size[0], bb_size[1], bb_size[2]])
    print(f"bb_size = {bb_size}")
    rods = RodComplexBC(h, meshes, transforms.numpy())
    viewer = PSViewer(rods)
    ps.set_user_callback(viewer.callback)
    ps.show()
    
if __name__ == "__main__":
    ps.init()
    ps.look_at((0, 4, 8), (0, 2, 0))
    # ps.set_ground_plane_mode("none")
    ps.set_ground_plane_height(-collision_eps)
    wp.config.max_unroll = 0
    wp.init()

    # multiple_drape()
    # drape()
    staggered_bars()
    # tets()
    # bunny_rain()
    # bar_rain()
    