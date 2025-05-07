import warp as wp
import numpy as np
from warp.sparse import bsr_mv
import polyscope as ps
import polyscope.imgui as gui
from mesh_complex import RodComplexBC
from medial_reduced import MedialRodComplex
from stretch import PSViewer, h, NewtonState, FEMMesh, compute_rhs, Triplets, add_dx
from g2m.viewer import MedialViewer
from g2m.fitter import SQEMFitter

@wp.func
def should_fix(xc: wp.vec3) -> bool:
    cyl0 = wp.vec3(0.64, .71, .64)
    cyl1 = wp.vec3(.64, 1.3, .64)
    elbow = wp.vec3(.64, 1.05, .64)
    length = 0.35
    r = 0.05

    y_in_bound = cyl0[1] - length / 2.0 < xc[1] < cyl0[1] + length / 2.0 or cyl1[1] - length / 2.0 < xc[1] < cyl1[1] + length / 2.0
    xz_projection_in_circle = wp.length_sq(wp.vec2(xc[0], xc[2]) - wp.vec2(.64, .64)) < r * r
    return y_in_bound and xz_projection_in_circle


@wp.func
def moving_boundary(xc: wp.vec3) -> bool: 
    cyl1 = wp.vec3(.64, 1.3, .64)
    length = 0.35
    r = 0.05
    y_in_bound = cyl1[1] - length / 2.0 < xc[1] < cyl1[1] + length / 2.0
    xz_projection_in_circle = wp.length_sq(wp.vec2(xc[0], xc[2]) - wp.vec2(.64, .64)) < r * r
    return y_in_bound and xz_projection_in_circle

@wp.kernel
def compute_compensation(state: NewtonState, geo: FEMMesh, theta: float, comp_x: wp.array(dtype = wp.vec3)):
    i = wp.tid()
    xi = state.x[i]
    c = wp.cos(theta)
    s = wp.sin(theta)
    
    pivot = wp.vec3(.64, 1.05, .64)
    rot = wp.mat22(
        c, s,
        -s, c
    )
    
    
    x_rst = geo.xcs[i]
    if moving_boundary(x_rst):
        yz_rst = wp.vec2(x_rst[1], x_rst[2]) - wp.vec2(pivot[1], pivot[2])
        yz = rot @ yz_rst
        yz += wp.vec2(pivot[1], pivot[2])
        target = wp.vec3(x_rst[0], yz[0], yz[1])
        comp_x[i] = xi - target
    else:
        comp_x[i] = wp.vec3(0.0)



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

@wp.kernel
def add_compx(state: NewtonState, comp_x: wp.array(dtype = wp.vec3)):
    i = wp.tid()
    state.x[i] -= comp_x[i]
        
class ArmBend(MedialRodComplex):
    def __init__(self, h, meshes = [], transforms = [], static_meshes = None):
        super().__init__(h, meshes, transforms, static_meshes)
        self.fitter = SQEMFitter(self.V, self.F, self.V_medial, self.R)
    
    def compute_rhs(self):
        wp.launch(compute_rhs, (self.n_nodes, ), inputs = [self.states, self.h, self.M, self.b])
        self.set_bc_fixed_grad()

        self.comp_x.zero_()
        wp.launch(compute_compensation, self.n_nodes, inputs= [self.states, self.geo, self.theta, self.comp_x])
        bsr_mv(self.A, self.comp_x, self.b, beta = 1.0)
        print(f"compensation = {np.linalg.norm(self.comp_x.numpy())}")

    def set_bc_fixed_hessian(self):
        wp.launch(set_K_fixed, (self.n_tets * 4 * 4,), inputs = [self.geo, self.triplets])

    def set_bc_fixed_grad(self):
        wp.launch(set_b_fixed, (self.n_nodes,), inputs = [self.geo, self.b])

    def line_search(self):
        # FIXME: not converged
        wp.launch(add_compx, dim = (self.n_nodes, ), inputs = [self.states, self.comp_x])
        wp.launch(set_b_fixed, dim = (self.n_nodes,), inputs = [self.geo, self.states.dx])
        return super().line_search()

    def get_VR_optimized(self):
        V, R = self.get_VR()
        self.fitter.V[:] = self.states.x.numpy()
        Vo, Ro = self.fitter.fit_spheres(V, R)
        return Vo, Ro
        
    # def process_collision(self):
    #     pass

    # def compute_collision_energy(self):
    #     return 0.0

class SQEMViewer(MedialViewer):
    def __init__(self, rod, static_mesh = None):
        super().__init__(rod, static_mesh)
        self.ui_use_sqem = False

    def callback(self):
        changed, self.ui_use_sqem = gui.Checkbox("SQEM", self.ui_use_sqem)
        super().callback()
    
    def update_medial(self):
        if self.ui_use_sqem:
            V, R  = self.rod.get_VR_optimized()
        else:
            V, R = self.rod.get_VR()
        self.V_medial[:] = V
        self.R[:] = R
        self.render_medial()

def arm_bend():
    meshes = ["assets/arm_bend/arm_bend.tobj"]
    transforms = np.array([np.identity(4, float)])
    arm = ArmBend(h, meshes, transforms)

    # viewer = PSViewer(arm)
    # viewer = MedialViewer(arm)
    viewer = SQEMViewer(arm)
    
    ps.set_user_callback(viewer.callback)
    ps.show()

if __name__ == "__main__":
    ps.init()
    ps.set_ground_plane_mode("none")
    ps.look_at((-1, 1, 0), (0, 1, 0))
    wp.config.max_unroll = 0
    wp.init()

    arm_bend()