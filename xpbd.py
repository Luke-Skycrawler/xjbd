import warp as wp 
import igl 

from params import *
@wp.struct
class Particle:
    x: wp.vec3
    x0: wp.vec3
    v: wp.vec3
    w: float
    # delta: wp.vec3
    # delta_count: int

@wp.struct
class Constraint:
    l0: float
    lam: float
    v0: int
    v1: int
    

@wp.kernel
def predict_position(p: wp.array(dtype = Particle)):
    i = wp.tid()
    p[i].x += p[i].v * dt

    if p[i].w > 0.0:
        p[i].v += dt * wp.vec3(0.0, -98.0, 0.0)


@wp.func
def id_flatten(i: int, j: int, resy: int) -> int: 
    return i * resy + j

@wp.kernel
def init_kernel(p: wp.array(dtype = Particle), constraints: wp.array(dtype = Constraint)):
    tid =  wp.tid()

    res = 40
    resy = 40
    i = tid // resy
    j = tid % resy

    p[tid].x = wp.vec3(float(i), float(j), 0.0)
    p[tid].v = wp.vec3(0.0, 0.0, 10.0 * float(resy - 1 - j) / float(resy))
    p[tid].x0 = p[tid].x


    pinned = (i == 0 and j == resy - 1) or (i == res - 1 and j == resy -1) 

    p[tid].w = wp.select(pinned, 1.0, 0.0)
    
    if i < res - 1:
        # horizontal constraints

        constraints[i + j * (res - 1)].v0 = id_flatten(i, j, resy)
        constraints[i + j * (res - 1)].v1 = id_flatten(i + 1, j, resy)

        constraints[i + j * (res - 1)].l0 = 1.0
    if j < resy - 1:
        # vertical constraints
        constraints[resy * (res - 1) + i + j * res].v0 = id_flatten(i, j, resy)
        constraints[resy * (res - 1) + i + j * res].v1 = id_flatten(i, j + 1, resy)

        constraints[resy * (res - 1) + i + j * res].l0 = 1.0
    
    

@wp.kernel
def strech_kernel(p: wp.array(dtype = Particle), constraints: wp.array(dtype = Constraint), deltas: wp.array(dtype = wp.vec3), delta_count: wp.array(dtype = int)):

    i = wp.tid()
    c = constraints[i]
    l0 = c.l0
    v10 = p[c.v0].x - p[c.v1].x
    dist = wp.length(v10)
    w0 = p[c.v0].w
    w1 = p[c.v1].w

    if w0 + w1 > 0.0:
        gradient = v10 / (dist + eps)
        denom = w0 + w1
        lam = (dist - l0) / denom
        common = lam *  gradient

        c0 = -w0 * common 
        c1 = w1 * common
        
        wp.atomic_add(deltas, c.v0 , c0)

        wp.atomic_add(deltas, c.v1, c1)
        wp.atomic_add(delta_count, c.v0, 1)

        wp.atomic_add(delta_count, c.v1, 1)

@wp.kernel
def add_dx_kernel(p: wp.array(dtype = Particle), deltas: wp.array(dtype = wp.vec3), delta_counts: wp.array(dtype = int)):
    i = wp.tid()
    if delta_counts[i] > 0:
        p[i].x += deltas[i] / float(delta_counts[i])
        deltas[i] = wp.vec3(0.0)
        delta_counts[i] = 0

@wp.kernel
def finalize_kernel(p: wp.array(dtype = Particle)):
    i = wp.tid()
    p[i].v = (p[i].x - p[i].x0) / dt
    p[i].x0 = p[i].x
    
class ClothGrid:
    def __init__ (self, res = 40):

        self.res = res
        self.shape = (res * res,)
        print(self.shape)
        self.particles = wp.zeros(self.shape, dtype = Particle)

        self.n_constraints = 2 * res * (res - 1)  
        self.constraints = wp.zeros((self.n_constraints), dtype = Constraint)

        self.deltas = wp.zeros(self.shape, dtype = wp.vec3)
        self.delta_counts = wp.zeros(self.shape, dtype = int)
        wp.launch(init_kernel, self.shape, inputs = [self.particles, self.constraints])


class ClothOBJ:
    def __init__(self, filename):
        V, _, _, F, _, _ = igl.read_obj(filename)
        self.shape = (V.shape[0], )
        E = igl.edges(F)
        self.n_constraints = E.shape[0]
        self.n_constraints = wp.zeros((self.n_constraints), dtype = Constraint)
        
        self.deltas = wp.zeros(self.shape, dtype = wp.vec3) 
        self.delta_counts = wp.zeros(self.shape, dtype = int)

class PBD(ClothGrid):
    def __init__(self, res = 40):
        super().__init__(res)
        self.n_iters = 5
        self.timer_active = True
        
    
    def step(self):
        with wp.ScopedTimer("pbd step"):
            self.predict_position()
            for _ in range(self.n_iters): 
                

                self.deltas.zero_()
                self.delta_counts.zero_()

                self.solve_strech()
                self.add_dx()
            self.finalize()

    def predict_position(self):
        wp.launch(predict_position, self.shape, inputs = [self.particles])

    def finalize(self):
        wp.launch(finalize_kernel, self.shape, inputs = [self.particles])


    def solve_strech(self):
        with wp.ScopedTimer("solve stretch", active = self.timer_active, synchronize= True):
            wp.launch(strech_kernel, (self.n_constraints,), inputs = [self.particles, self.constraints, self.deltas, self.delta_counts])

    def add_dx(self):
        wp.launch(add_dx_kernel, self.shape, inputs = [self.particles, self.deltas, self.delta_counts])


if __name__ == "__main__":
    wp.init()
    import polyscope as ps 
    ps.init()
    sim = PBD(40)

    get_x = lambda sim: sim.particles.numpy()["x"].reshape((-1, 3))
    cloud = ps.register_point_cloud("particles", get_x(sim))
    
    while(True):
        sim.step()
        xnp = get_x(sim)
        cloud.update_point_positions(xnp)    
        ps.frame_tick()