import polyscope as ps
import polyscope.imgui as gui 
import numpy as np
import warp as wp 
from fem.interface import Rod, default_tobj, RodComplex, TOBJComplex
from fem.params import model
import igl
from warp.sparse import *
from fem.params import FEMMesh, mu, lam, gravity, gravity_np
from fem.fem import tet_kernel, tet_kernel_sparse, Triplets, psi
from warp.optim.linear import bicgstab, cg

eps = 3e-4
h = 2e-3
rho = 1e3
omega = 3.0
@wp.struct 
class NewtonState: 
    x: wp.array(dtype = wp.vec3)
    x0: wp.array(dtype = wp.vec3)
    dx: wp.array(dtype = wp.vec3)
    xdot: wp.array(dtype = wp.vec3)
    M: wp.array(dtype = float)
    Psi: wp.array(dtype = float)

@wp.kernel
def set_M_diag(d: wp.array(dtype = float), M: wp.array(dtype = wp.mat33)):  
    i =  wp.tid()
    mii = wp.identity(3, dtype = float)
    mii *= d[i]
    M[i] = mii


@wp.func
def x_minus_tilde(state: NewtonState, h: float, i: int) -> wp.vec3:
    return state.x[i] - (state.x0[i] + h * state.xdot[i] + h * h * gravity)

@wp.kernel
def compute_rhs(state: NewtonState, h: float, M: wp.array(dtype = float), b: wp.array(dtype = wp.vec3)):
    i = wp.tid()
    b[i] = -b[i] * h * h + M[i] * x_minus_tilde(state, h, i)


@wp.func
def should_fix(x: wp.vec3): 
    return x[0] < -0.5 + eps# or x[0] > 0.5 - eps

    # v0 = wp.vec3(-56.273449910216, 94.689259419722, -19.03583034376)
    # return wp.length_sq(x - v0) < eps
@wp.func
def moving_boundary(x: wp.vec3):
    return x[0] < -0.5 + eps# or x[0] > 0.5 - eps
    

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
def add_dx(state: NewtonState, alpha :float):
    i = wp.tid()
    state.x[i] -= state.dx[i] * alpha

@wp.kernel
def update_x0_xdot(state: NewtonState, h: float):
    i = wp.tid()
    state.xdot[i] = (state.x[i] - state.x0[i]) / h
    state.x0[i] = state.x[i]

@wp.kernel
def compute_Psi(x: wp.array(dtype = wp.vec3), geo: FEMMesh, Bm: wp.array(dtype = wp.mat33), W: wp.array(dtype = float), Psi: wp.array(dtype = float)):
    e = wp.tid()
    t0 = x[geo.T[e, 0]]
    t1 = x[geo.T[e, 1]]
    t2 = x[geo.T[e, 2]]
    t3 = x[geo.T[e, 3]]
    
    Ds = wp.mat33(t0 - t3, t1 - t3, t2 - t3)
    
    F = Ds @ Bm[e]
    psie = psi(F)
    # wp.atomic_add(Psi, 0, W[e] * psi)
    Psi[e] = W[e] * psie

@wp.kernel
def compute_inertia(state: NewtonState, M: wp.array(dtype = float), inert: wp.array(dtype = float), h: float):
    i = wp.tid()
    dx = x_minus_tilde(state, h, i)
    de = wp.length_sq(dx) * M[i] * 0.5
    wp.atomic_add(inert, 0, de)

@wp.kernel
def compute_compensation(state: NewtonState, geo: FEMMesh, theta: float, comp_x: wp.array(dtype = wp.vec3)):
    i = wp.tid()
    xi = state.x[i]
    c = wp.cos(theta)
    s = wp.sin(theta)
    rot = wp.mat22(
        c, s,
        -s, c
    )
    if moving_boundary(xi):
    # if False:
        x_rst= geo.xcs[i]
        yz_rst = wp.vec2(x_rst[1], x_rst[2])
        yz = rot @ yz_rst
        if x_rst[0] < 0.0:
            yz = wp.transpose(rot) @ yz_rst
        target = wp.vec3(x_rst[0], yz[0], yz[1]) # + wp.vec3(0.0, theta, 0.0)
        # target = x_rst + wp.vec3(0.0, theta, 0.0)
        comp_x[i] = xi - target
    else:
        comp_x[i] = wp.vec3(0.0)

class PSViewer:
    def __init__(self, rod, static_mesh: TOBJComplex = None):
        self.V = rod.xcs.numpy()
        self.F = rod.F

        self.ps_mesh = ps.register_surface_mesh("rod", self.V, self.F)
        self.frame = 0
        self.rod = rod
        self.ui_pause = True
        self.animate = False
        if static_mesh is not None:
            Vs = static_mesh.xcs.numpy()
            Fs = static_mesh.indices.numpy().reshape((-1, 3))
            self.static_mesh = ps.register_surface_mesh("static", Vs, Fs)

    def callback(self):
        changed, self.ui_pause = gui.Checkbox("Pause", self.ui_pause)
        self.animate = gui.Button("Step") or not self.ui_pause
        if gui.Button("Reset"):
            self.rod.reset()
            self.frame = 0
            self.ui_pause = True
            self.animate = True

        if gui.Button("Save"):
            np.save(f"output/x_{self.frame}.npy", self.V)
            print(f"output/x_{self.frame}.npy saved")
        if self.animate: 
            self.rod.step()
            self.V = self.rod.states.x.numpy()
            self.ps_mesh.update_vertex_positions(self.V)
            self.frame += 1
            
            print("frame = ", self.frame)

            # ps.screenshot(f"output/{self.frame:04d}.jpg")

        
class RodBCBase:
    '''
    fem with boundary condition and dynamic attributes
    '''

    def  __init__(self, h):
        super().__init__()
        self.define_M()
        self.states = NewtonState()
        self.states.x = wp.zeros_like(self.xcs)
        self.states.x0 = wp.zeros_like(self.xcs)
        self.states.dx = wp.zeros_like(self.xcs)
        self.states.xdot = wp.zeros_like(self.xcs)
        self.states.Psi = wp.zeros((self.n_tets,), dtype = float)

        self.comp_x = wp.zeros_like(self.states.dx)
        
        self.reset()
        self.h = h
        print(f"timestep set to {h}")
        
    def reset(self):
        wp.copy(self.states.x, self.xcs)
        wp.copy(self.states.x0, self.xcs)

        self.theta = 0.0

    def define_M(self):
        V = self.xcs.numpy()
        T = self.T.numpy()
        # self.M is a vector composed of diagonal elements 
        self.Mnp = igl.massmatrix(V, T, igl.MASSMATRIX_TYPE_BARYCENTRIC).diagonal()
        self.M = wp.zeros((self.n_nodes,), dtype = float)
        self.M.assign(self.Mnp * rho)

        self.M_sparse = bsr_zeros(self.n_nodes, self.n_nodes, wp.mat33)
        M_diag = wp.zeros((self.n_nodes,), dtype = wp.mat33)
        wp.launch(set_M_diag, (self.n_nodes,), inputs = [self.M, M_diag])
        bsr_set_diag(self.M_sparse, M_diag)


    def step(self):
        newton_iter = True
        n_iter = 0
        max_iter = 2
        # while n_iter < max_iter:
        while newton_iter:
            self.compute_A()
            self.compute_rhs()

            self.solve()
            # wp.launch(add_dx, dim = (self.n_nodes, ), inputs = [self.states, 1.0])
            
            
            # line search stuff, not converged yet
            alpha = self.line_search()
            if alpha == 0.0:
                break

            dxnp = self.states.dx.numpy()
            norm_dx = np.linalg.norm(dxnp)
            newton_iter = norm_dx > 1e-3 and n_iter < max_iter
            print(f"norm = {np.linalg.norm(dxnp)}, {n_iter}")
            n_iter += 1
        self.update_x0_xdot()
        self.theta += self.h * omega

    def update_x0_xdot(self):
        wp.launch(update_x0_xdot, dim = (self.n_nodes,), inputs = [self.states, self.h])

    def compute_A(self):

        self.compute_K()

        # A = h^2 * K + M
        h = self.h
        bsr_axpy(self.M_sparse, self.K_sparse, 1.0, h * h)
        self.A = self.K_sparse

    def compute_K(self):
        self.triplets.vals.zero_()
        self.b.zero_()
        wp.launch(tet_kernel_sparse, (self.n_tets * 4 * 4,), inputs = [self.states.x, self.geo, self.Bm, self.W, self.triplets, self.b]) 
        # now self.b has the elastic forces

        self.set_bc_fixed_hessian()
        bsr_set_zero(self.K_sparse)
        bsr_set_from_triplets(self.K_sparse, self.triplets.rows, self.triplets.cols, self.triplets.vals)        
        
    def compute_rhs(self):
        wp.launch(compute_rhs, (self.n_nodes, ), inputs = [self.states, self.h, self.M, self.b])
        self.set_bc_fixed_grad()
        # self.comp_x.zero_()
        # wp.launch(compute_compensation, self.n_nodes, inputs= [self.states, self.geo, self.theta, self.comp_x])
        # bsr_mv(self.A, self.comp_x, self.b, beta = 1.0)
        # print(f"compensation = {np.linalg.norm(self.comp_x.numpy())}")

    def set_bc_fixed_grad(self):
        wp.launch(set_b_fixed, (self.n_nodes,), inputs = [self.geo, self.b])
    
    def set_bc_fixed_hessian(self):
        wp.launch(set_K_fixed, (self.n_tets * 4 * 4,), inputs = [self.geo, self.triplets])

    def solve(self):
        with wp.ScopedTimer("solve"):
            self.states.dx.zero_()
            # bicgstab(self.A, self.b, self.states.dx, 1e-6, maxiter = 100)
            cg(self.A, self.b, self.states.dx, 1e-4, use_cuda_graph = True)
    
    # def line_search(self):
    #     alpha = 1.0
    #     wp.launch(add_dx, dim = (self.n_nodes, ), inputs = [self.states, alpha])
    #     return alpha
        
    def line_search(self):
        # FIXME: not converged
        x_tmp = wp.clone(self.states.x)
        E0 = self.compute_psi() + self.compute_inertia() + self.compute_collision_energy()
        alpha = 1.0
        while True:
            wp.copy(self.states.x, x_tmp)
            wp.launch(add_dx, dim = (self.n_nodes, ), inputs = [self.states, alpha])
            E1 = self.compute_psi() + self.compute_inertia() + self.compute_collision_energy()
            
            if E1 < E0:
                break
            if alpha < 1e-3:
                wp.copy(self.states.x, x_tmp)
                alpha = 0.0
                break
            alpha *= 0.5

        # print(f"alpha = {alpha}, E0 = {E0}, E1 = {E1}")
        return alpha

    def compute_collision_energy(self):
        return 0.0

    def compute_psi(self):
        h = self.h
        self.states.Psi.zero_()
        wp.launch(compute_Psi, (self.n_tets,), inputs = [self.states.x, self.geo, self.Bm, self.W, self.states.Psi])
        return np.sum(self.states.Psi.numpy()) * h * h
    
    def compute_inertia(self):
        inert = wp.zeros((1,), dtype = float)
        wp.launch(compute_inertia, (self.n_nodes, ), inputs = [self.states, self.M, inert, self.h])
        return inert.numpy()[0]

class RodBC(RodBCBase, Rod):
    def __init__(self, h, filename = default_tobj):
        self.filename = filename
        super().__init__(h)

def drape():
    # rod = RodBC(h, "assets/elephant.mesh")
    rod = RodBC(h)
    viewer = PSViewer(rod)
    ps.set_user_callback(viewer.callback)
    ps.show()

def twist():
    rod = RodBC(h, "assets/bar2.tobj")
    viewer = PSViewer(rod)
    ps.set_user_callback(viewer.callback)
    ps.show()

if __name__ == "__main__":
    ps.init()
    wp.init()
    # drape()
    twist()
    # multiple_drape()
    