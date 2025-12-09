import warp as wp
from stretch import RodBCBase, PSViewer, NewtonState, Triplets, add_dx, h, eps, FEMMesh 
from fem.interface import Rod, RodComplex
import polyscope as ps
import numpy as np
# collision add-ons
from geometry.collision_cell import MeshCollisionDetector, collision_eps
from utils.tobj import import_tobj
from warp.sparse import bsr_axpy, bsr_set_from_triplets, bsr_zeros
from geometry.static_scene import StaticScene

omega = 3.0

@wp.kernel
def init_velocities(states: NewtonState, positions: wp.array(dtype = wp.vec3), n_verts: int):
    i = wp.tid()
    j = i // n_verts
    pi = positions[j]
    v = wp.vec3(-pi[0], 0.0, -pi[2])
    states.xdot[i] = v * 0.5

@wp.func
def should_fix(x: wp.vec3): 
    return wp.abs(x[0]) < eps * 10. and wp.abs(x[1]) < eps * 10.

@wp.kernel
def set_b_fixed(geo: FEMMesh,b: wp.array(dtype = wp.vec3)):
    i = wp.tid()
    # set fixed points rhs to 0
    if should_fix(geo.xcs[i]): 
        b[i] = wp.vec3(0.0, 0.0, 0.0)

@wp.kernel
def set_K_fixed(geo: FEMMesh, triplets: Triplets):
    eij = wp.tid()
    e = eij // 16
    ii = (eij // 4) % 4
    jj = eij % 4

    i = geo.T[e, ii]
    j = geo.T[e, jj]
    
    if should_fix(geo.xcs[i]) or should_fix(geo.xcs[j]):        
        if ii == jj:
            triplets.vals[eij] = wp.identity(3, dtype = float)
        else:
            triplets.vals[eij] = wp.mat33(0.0)

class Timeit:
    def __init__(self):
        self.dict = {
            "compute A": [],
            "collision": [],              
            "assemble": [],
            "solve": [],
            "line search": [],
            "detection": [],
        }
        
    def reset(self):
        for k in self.dict.keys():
            self.dict[k] = 0.0


class RodComplexBC(RodBCBase, RodComplex):
    def __init__(self, h, meshes = [], transforms = [], static_meshes:StaticScene = None):
        self.meshes_filename = meshes 
        self.transforms = transforms
        super().__init__(h)
        self.static_meshes = static_meshes
        self.define_collider()
        self.n_pt = 0
        self.n_ee = 0
        self.n_ground = 0

        self.timeit = Timeit()
        self.tot_iters = 0
        

    def define_collider(self):
        self.collider = MeshCollisionDetector(self.states.x, self.T, self.indices, self.Bm, ground = 0.0, static_objects = self.static_meshes)

    def reset(self):
        n_verts = 4
        self.theta = 0.0
        self.frame = 0
        model = self.meshes_filename[0].split("/")[1].split(".")[0]
        model_ntets = {
            "bar2": 525,
            "tet": 4,
            "bunny_5": 1356,
            "bug": 2471,
            "squishy": 4778,
            "bunny": 3679, 
            "windmill": 1350
        }
        n_verts = model_ntets[model]
        wp.copy(self.states.x, self.xcs)
        wp.copy(self.states.x0, self.xcs)
        if "assets/tet.tobj" in self.meshes_filename:
            wp.launch(set_vx_kernel, (self.n_nodes,), inputs = [self.states, n_verts])
        elif model in["bar2", "bug", "squishy", "bunny", "windmill"]: 
            # wp.launch(set_velocity_kernel, (self.n_nodes,), inputs = [self.states, n_verts])
            pass
        else: 
            pos = self.transforms[:, :3, 3]
            positions = wp.array(pos, dtype = wp.vec3)
            wp.launch(init_velocities, (self.n_nodes,), inputs = [self.states, positions, n_verts])
        

    def set_bc_fixed_hessian(self):
        pass
        # wp.launch(set_K_fixed, (self.n_tets * 4 * 4,), inputs = [self.geo, self.triplets])

    def set_bc_fixed_grad(self):
        pass
        # wp.launch(set_b_fixed, (self.n_nodes,), inputs = [self.geo, self.b])
    
    def process_collision(self):
        with wp.ScopedTimer("collision"):
            with wp.ScopedTimer("hess & grad"):
                triplets = self.collider.analyze(self.b, self.n_pt, self.n_ee, self.n_ground)
                # triplets = self.collider.analyze(self.b)
            with wp.ScopedTimer("build_from_triplets"):
                self.add_collision_to_sys_matrix(triplets)
    
    def line_search_fixed(self):
        return 1.0

    def cd(self):
        self.n_pt, self.n_ee, self.n_ground = self.collider.collision_set("all") 

    def step(self):
        self.theta += omega * self.h
        with wp.ScopedTimer("step"):
            newton_iter = True
            self.n_iter = 0
            max_iter = 20
            # while n_iter < max_iter:
            while newton_iter:
                with wp.ScopedTimer(f"newton #{self.n_iter}"):
                    with wp.ScopedTimer("compute A", dict = self.timeit.dict):
                        self.compute_A()

                    with wp.ScopedTimer("detection", dict = self.timeit.dict):
                        self.cd()
                    with wp.ScopedTimer("collision", dict = self.timeit.dict):
                        self.process_collision()
                    
                    with wp.ScopedTimer("assemble", dict = self.timeit.dict):
                        self.compute_rhs()

                    with wp.ScopedTimer("solve", dict = self.timeit.dict):
                        self.solve()
                    # wp.launch(add_dx, dim = (self.n_nodes, ), inputs = [self.states, 1.0])
                    self.n_iter += 1
                    
                    newton_iter = not (self.converged() or self.n_iter >= max_iter)
                    if not newton_iter:
                        alpha = self.line_search_fixed()
                        print(f"\niter = {self.n_iter}, alpha = {alpha}, dz norm = {self.norm_dz:.2e}\n")
                        break
                    # line search stuff, not converged yet
                    with wp.ScopedTimer("line search", dict = self.timeit.dict):
                        alpha = self.line_search()
                    if alpha == 0.0:
                        print("\nline search failed")
                        break

                    print(f"\niter = {self.n_iter}, alpha = {alpha}, dz norm = {self.norm_dz:.2e}\n")
            self.update_x0_xdot()
            self.frame += 1
            
    def save_meta(self):
        print(f"total iterations = {self.tot_iters}, total frames = {self.frame}")
        np.save("plot/metadata.npy", {"total_iters": self.tot_iters, "total_frames": self.frame})
        # np.save("timeit.npy", self.timeit.dict)
        np.savez("plot/timeit.npz", **self.timeit.dict)

    def converged(self):
        dxnp = self.states.dx.numpy()
        norm_dx = np.linalg.norm(dxnp)
        print(f"norm dx = {np.linalg.norm(dxnp)}")
        return norm_dx < 1e-3
            
    def add_collision_to_sys_matrix(self, triplets: Triplets):

        collision_force_derivatives = bsr_zeros(self.n_nodes, self.n_nodes, wp.mat33)
        bsr_set_from_triplets(collision_force_derivatives, triplets.rows, triplets.cols, triplets.vals)
        bsr_axpy(collision_force_derivatives, self.A, self.h * self.h)

    def compute_collision_energy(self):
        self.n_pt, self.n_ee, self.n_ground = self.collider.collision_set("all")
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
    states.xdot[i] = wp.vec3(0.0, -1.0, 0.0)
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
        pos[dim] = float(xd) / float(grid_dims[dim]) * 30.0 - 15.0
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
    

def staggered_bug():
    model = "bunny"
    # model = "bug"
    n_meshes = 1
    meshes = [f"assets/{model}/{model}.tobj"] * n_meshes
    transforms = [np.identity(4, dtype = float) for _ in range(n_meshes)]

    transforms[-1][:3, :3] = np.zeros((3, 3))
    transforms[-1][0, 1] = 1.5
    transforms[-1][1, 0] = 1.5
    transforms[-1][2, 2] = 1.5

    for i in range(n_meshes):
        # transforms[i][0, 3] = i * 0.5
        transforms[i][1, 3] = 1.2 + i * 0.25
        transforms[i][2, 3] = i * 1.2 - 0.8
    
    # rods = MedialRodComplex(h, meshes, transforms)

    # scale params for teapot
    static_meshes_file = ["assets/teapotContainer.obj"]
    scale = np.identity(4) * 3
    scale[3, 3] = 1.0

    # bouncy box
    static_meshes_file = ["assets/bouncybox.obj"]
    box_size = 4
    scale = np.identity(4) * box_size
    scale[3, 3] = 1.0
    scale[:3, 3] = np.array([0, box_size, box_size / 2], float)
    for i in range(n_meshes):
        transforms[i][1, 3] += box_size * 1.5
        
    
    static_bars = StaticScene(static_meshes_file, np.array([scale]))
    # static_bars = None
    rods = RodComplexBC(h, meshes, transforms, static_bars)
    
    
    viewer = PSViewer(rods, static_bars)
    ps.set_user_callback(viewer.callback)
    ps.show()
def windmill():
    # model = "bunny"
    model = "windmill"
    # model = "bug"
    n_meshes = 1
    meshes = [f"assets/{model}/{model}.tobj"] * n_meshes
    transforms = [np.identity(4, dtype = float) for _ in range(n_meshes)]

    transforms[-1][:3, :3] = np.zeros((3, 3))
    transforms[-1][0, 0] = 0.5
    transforms[-1][2, 1] = 0.5
    transforms[-1][1, 2] = 0.5
    # transforms[-1][0, 1] = 1.5
    # transforms[-1][1, 0] = 1.5
    # transforms[-1][2, 2] = 1.5

    # for i in range(n_meshes):
    #     # transforms[i][0, 3] = i * 0.5
    #     transforms[i][1, 3] = 1.2 + i * 0.25
    #     transforms[i][2, 3] = i * 1.2 - 0.8
    
    # rods = MedialRodComplex(h, meshes, transforms)

    # scale params for teapot
    static_meshes_file = ["assets/teapotContainer.obj"]
    scale = np.identity(4) * 3
    scale[3, 3] = 1.0

    # bouncy box
    # static_meshes_file = ["assets/bouncybox.obj"]
    # box_size = 4
    # scale = np.identity(4) * box_size
    # scale[3, 3] = 1.0
    # scale[:3, 3] = np.array([0, box_size, box_size / 2], float)
    # for i in range(n_meshes):
    #     transforms[i][1, 3] += box_size * 1.5
        
    

    # static_bars = StaticScene(static_meshes_file, np.array([scale]))
    static_bars = None
    # rods = MedialRodComplex(h, meshes, transforms, static_bars)
    rods = RodComplexBC(h, meshes, transforms, static_bars)
    
    
    viewer = PSViewer(rods, static_bars)
    ps.set_user_callback(viewer.callback)
    ps.show()

if __name__ == "__main__":
    ps.init()
    ps.look_at((0, 4, 10), (0, 4, 0))
    # ps.set_ground_plane_mode("none")
    ps.set_ground_plane_height(-collision_eps)
    wp.config.max_unroll = 0
    wp.init()

    # multiple_drape()
    # drape()
    # staggered_bars()
    # staggered_bug()
    windmill()
    # tets()
    # bunny_rain()
    # bar_rain()
    