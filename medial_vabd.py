import warp as wp
import numpy as np
import polyscope as ps 
import polyscope.imgui as gui

from stretch import h, add_dx, compute_rhs
from medial_reduced import MedialRodComplex, vec

from scipy.linalg import lu_factor, lu_solve, solve, polar
from ortho import OrthogonalEnergy
from g2m.viewer import MedialViewer
from vabd import per_node_forces
ad_hoc = True
medial_collision_stiffness = 5e3
def asym(a):
    return 0.5 * (a - a.T)
class MedialVABD(MedialRodComplex):
    def __init__(self, h, meshes=[], transforms=[]):
        super().__init__(h, meshes, transforms)
        self.split_U0_U_tilide()
        self.define_mm()
        self.prefactor_once()
        self.define_A0_b0_btilde()
        
        self.ortho = OrthogonalEnergy()
        self.sum_weights()

    def sum_weights(self):
        ww  = self.W.numpy()
        n_nodes_per_mesh = self.n_nodes // self.n_meshes
        weights = ww.reshape((self.n_meshes, -1))
        self.sum_W = np.sum(weights, axis = 1, keepdims = False)

    def define_A0_b0_btilde(self):
        self.A0 = np.zeros((self.n_meshes * 12, self.n_meshes * 12))
        self.b0 = np.zeros((self.n_meshes * 12,))
        self.b_tilde = np.zeros((self.n_meshes * (self.n_modes - 12),))

    def split_U0_U_tilide(self):
        self.U0 = np.zeros((self.n_nodes * 3, self.n_meshes * 12))
        self.U_tilde = np.zeros((self.n_nodes * 3, (self.n_modes - 12) * self.n_meshes))
        n_nodes_per_mesh = self.n_nodes // self.n_meshes
        x0np = self.xcs.numpy()
        for i in range(self.n_meshes):
            start = i * n_nodes_per_mesh * 3
            end = start + n_nodes_per_mesh * 3
            
            V = x0np[i * n_nodes_per_mesh: (i + 1) * n_nodes_per_mesh]
            w0 = self.Q[:, 0: 1]
            
            rhs = self.lbs_matrix(V, w0)
            self.U0[start: end, 12 * i: 12 * (i + 1)] = rhs


            self.U_tilde[start:end, i * (self.n_modes - 12): (i + 1) * (self.n_modes - 12)] = self.lbs_matrix(V, self.Q[:, 1:])

        # fill Um_tilde 
        self.Um_tilde = np.zeros((self.n_medial * 3, self.z_tilde.shape[0]))
        self.Um0 = np.zeros((self.n_medial * 3, 12 * self.n_meshes))
        for i in range(self.n_meshes):
            start = i * self.n_modes
            end = (i + 1) * self.n_modes

            self.Um_tilde[:, i * (self.n_modes - 12): (i + 1) * (self.n_modes - 12)] = self.Um[:, start + 12: end]
            self.Um0[:, i * 12: (i + 1) * 12] = self.Um[:, start: start + 12]

    def define_mm(self):
        self.mm = self.U0.T @ self.to_scipy_bsr(self.M_sparse) @ self.U0
    
    
    def get_F(self, i):
        '''
        columns of F is formed by segments of z 
        vec(F) = z
        '''
        z = self.z[i * self.n_modes: i * self.n_modes + 12]
        F = z.reshape((-1, 3)).T
        return F[:, :3]

    def prefactor_once(self):
        h = self.h
        self.compute_K()
        self.K0 = self.U_tilde.T @ self.to_scipy_bsr() @ self.U_tilde * (h * h)
        self.M_tilde = self.U_tilde.T @ self.to_scipy_bsr(self.M_sparse) @ self.U_tilde

        self.A_tilde = self.K0 + self.M_tilde

        # cholesky factorization tend to have non-positive definite matrix, use lu instead
        self.c, self.low = lu_factor(self.A_tilde)
        

    def define_z(self, transforms):
        self.n_modes = self.Q.shape[1] * 12 
        self.n_meshes = len(transforms)
        self.n_reduced = self.n_modes * self.n_meshes
        
        self.z = np.zeros(self.n_reduced)
        for i in range(self.n_meshes):
            self.z[i * self.n_modes: i * self.n_modes + 9] = vec(np.identity(3))

        self.z0 = np.zeros(self.n_meshes * 12)
        self.z_dot = np.zeros_like(self.z0)

        self.dz = np.zeros_like(self.z)

        # define z_tilde
        self.z_tilde = np.zeros(self.n_reduced - 12 * self.n_meshes)
        self.z_tilde0 = np.copy(self.z_tilde)
        self.z_tilde_dot = np.zeros_like(self.z_tilde)
        self.dz_tilde = np.zeros_like(self.z_tilde)

    def extract_z0(self, z):
        z0 = np.zeros(self.n_meshes * 12)
        for i in range(self.n_meshes):
            start = i * self.n_modes
            end = start + 12
            z0[i * 12: (i + 1) * 12] = self.z[start: end]
        return z0

    def update_x0_xdot(self):
        # super().update_x0_xdot()

        z = self.extract_z0(self.z)
        self.z_dot[:] = (z - self.z0) / self.h
        self.z0[:] = z

        self.z_tilde_dot[:] = (self.z_tilde - self.z_tilde0) / self.h
        self.z_tilde0[:] = self.z_tilde

        self.states.x.assign((self.U @ self.z).reshape((-1, 3)))

    def dz_tiled2dz(self, dz_tilde, dz0):
        dz = np.zeros_like(dz_tilde)

        for i in range(self.n_meshes):
            F = self.get_F(i)
            R, p = polar(F)
            
            start = i * (self.n_modes - 12)
            end = start + (self.n_modes - 12)

            dzti = dz_tilde[start: end].reshape((-1, 3)).T
            dz0i = dz0[i * 12: (i + 1) * 12]
            dR = asym(dz0i.reshape((-1, 3)).T[:, :3])


            zti = self.z_tilde[start: end].reshape((-1, 3)).T

            dz[start: end] = (R @ dzti + dR @ zti).T.reshape(-1)

        return dz

    # super.solve:
    # def solve(self):
    #     dz = solve(self.A_reduced, self.b_reduced, assume_a="sym")
    #     self.dz[:] = dz
    #     self.states.dx.assign((self.U @ dz).reshape(-1, 3))

    def solve(self):
        self.A0[:] = 0.
        self.b0[:] = 0.
        self.b_tilde[:] = 0.

        h2 = self.h * self.h
        
        # set A0, b0
        for i in range(self.n_meshes):
            fi = self.get_F(i)

            g, H = self.ortho.analyze(fi)
            mmi = self.mm[i * 12: (i + 1) * 12, i * 12: (i + 1) * 12]
            self.A0[i * 12: (i + 1) * 12, i * 12: (i + 1) * 12] = H * h2 * self.sum_W[i] + mmi
            self.b0[i * 12: (i + 1) * 12] = g * h2 * self.sum_W[i] + mmi @ (self.z[i * self.n_modes: i * self.n_modes + 12] - self.z_hat(i))

        # set b_tilde
        self.b_tilde = self.K0 @ self.z_tilde + self.M_tilde @ (self.z_tilde - self.z_tilde_hat()) + self.compute_excitement()

        # dz0 = solve(self.A0, self.b0, assume_a="sym")
        # dz0 = solve(self.A0 + self.A0_col, self.b0 + self.b0_col, assume_a="sym")

        
        # self.dz_tilde = lu_solve((self.c, self.low), self.b_tilde)

        A_sys = np.zeros((self.n_reduced, self.n_reduced))
        b_sys = np.zeros(self.n_reduced)
        


        # U_prime = self.compute_U_prime()
        # U_tildeT = U_prime @ self.U_tilde.T
        # U_sys = np.vstack([self.U0.T, U_tildeT])

        # A_sys = U_sys @ self.to_scipy_bsr() @ U_sys.T 
        # b_sys = U_sys @ self.b.numpy().reshape(-1) 
        
        A_sys[:self.n_meshes * 12, :self.n_meshes * 12] = self.A0# + self.A0_col
        A_sys[:self.n_meshes * 12, self.n_meshes * 12:] = 0.0
        A_sys[self.n_meshes * 12:, :self.n_meshes * 12] = 0.0
        # np.save("A_tilde.npy", A_sys[self.n_meshes * 12:, self.n_meshes * 12:])
        # np.save("A_tilde_K0.npy", self.A_tilde)
        A_sys[self.n_meshes * 12:, self.n_meshes * 12:] = self.A_tilde# + self.A_col_tilde

        b_sys[:self.n_meshes * 12] = self.b0# + self.b0_col
        b_sys[self.n_meshes * 12:] = self.b_tilde# + self.b_col_tilde

        dz_sys = solve(A_sys + self.A_sys_col, b_sys + self.b_sys_col, assume_a="sym")
        # dz_sys = solve(A_sys, b_sys, assume_a="sym")
        
        
        dz0 = dz_sys[:self.n_meshes * 12]
        self.dz_tilde[:] = dz_sys[self.n_meshes * 12:]

        dz = np.zeros_like(self.z)
        dz_from_zt = self.dz_tiled2dz(self.dz_tilde, dz0)
        for i in range(self.n_meshes):
            start = i * self.n_modes
            end = start + self.n_modes
            dz[start: start + 12] = dz0[i * 12: (i + 1) * 12]
            dz[start + 12: end] = dz_from_zt[i * (self.n_modes - 12): (i + 1) * (self.n_modes - 12)]

        self.dz[:] = dz
        self.states.dx.assign((self.U @ dz).reshape(-1, 3))

    def line_search(self):
        z_tilde_tmp = np.copy(self.z_tilde)
        z_tmp = np.copy(self.z)

        alpha = 1.0
        E0 = self.compute_psi() + self.compute_inertia() + self.compute_collision_energy()
        while True:
            self.z_tilde[:] = z_tilde_tmp - alpha * self.dz_tilde
            self.z[:] = z_tmp - alpha * self.dz
            E1 = self.compute_psi() + self.compute_inertia() + self.compute_collision_energy()
            if E1 < E0:
                break
            if alpha < 1e-3:
                self.z_tilde[:] = z_tilde_tmp
                self.z[:] = z_tmp
                alpha = 0.0
                break
            alpha *= 0.5
        print(f"line search alpha = {alpha}")
        return alpha
    
    def compute_psi(self):
        return 0.

    def compute_inertia(self):
        h2 = self.h ** 2
        ret = 0.0
        for i in range(self.n_meshes):
            fi = self.get_F(i)
            e = self.ortho.energy(fi)
            mmi = self.mm[i * 12: (i + 1) * 12, i * 12: (i + 1) * 12]

            dz = self.z[i * self.n_modes: i * self.n_modes + 12] - self.z_hat(i)
            ret += e * h2 * self.sum_W[i] + 0.5 * dz @ mmi @ dz

        dzt = self.z_tilde - self.z_tilde_hat()
        ret += (self.z_tilde @ self.K0 @ self.z_tilde + dzt @ self.M_tilde @ dzt) * 0.5
        return ret

    def compute_collision_energy(self):
        V, R = self.get_VR()
        h = self.h
        return self.collider_medial.energy(V, R) * medial_collision_stiffness * h * h    

    # def line_search(self):
    #     self.z_tilde -= self.dz_tilde
    #     self.z -= self.dz
        
    #     # self.states.x.assign((self.U @ self.z).reshape((-1, 3)))
    #     return 1.0
        
    def z_tilde_hat(self):
        return self.h * self.z_tilde_dot + self.z_tilde0

    def z_hat(self, i):
        zh = self.z0 + self.h * self.z_dot
        return zh[i * 12: (i + 1) * 12]

    def compute_U_prime(self):
        U_prime_dim = self.z_tilde.shape[0]
        U_prime = np.zeros((U_prime_dim, U_prime_dim))
        for i in range(self.n_meshes):
            fi = self.get_F(i)
            ri, _ = polar(fi)
            
            i_m = np.identity((self.n_modes - 12) // 3, float)
            start  = i * (self.n_modes - 12)
            end = start + (self.n_modes - 12)
            U_prime[start:end, start: end] = np.kron(i_m, ri)
        
        return U_prime

    def compute_excitement(self):
        return np.zeros_like(self.z_tilde)
        f = self.per_node_forces() * 2.0
        U_prime = self.compute_U_prime()        
        Q_tilde = U_prime @ self.U_tilde.T @ f
        return Q_tilde
    
    def per_node_forces(self):
        b = wp.zeros_like(self.b)
        wp.launch(per_node_forces, (self.n_nodes, ), inputs = [self.geo, b, self.h])
        return b.numpy().reshape(-1)

    def reset_z(self):
        self.z[:] = 0.0
        self.dz[:] = 0.0

        
        for i in range(self.n_meshes):
            self.z[i * self.n_modes: i * self.n_modes + 9] = vec(np.identity(3))
        self.z0[:] = self.extract_z0(self.z)
        self.z_dot[:] = 0.0
        
        if ad_hoc: 
            self.z_dot[12 + 9: 12 + 12] = np.array([0.0, 0.0, -3.0])
            
            # self.z_dot[2] = -1.0
            # self.z_dot[6] = 1.0
        
        self.z_tilde[:] = 0.0
        self.z_tilde0[:] = 0.0
        self.z_tilde_dot[:] = 0.0
        self.dz_tilde[:] = 0.0

        self.frame = 0
    def process_collision(self):
        V, R = self.get_VR()
        with wp.ScopedTimer("detect"):
            self.collider_medial.collision_set(V, R)
        with wp.ScopedTimer("analyze"):
            g, H = self.collider_medial.analyze()
        U_prime = self.compute_U_prime()
        Um_tildeT = U_prime @ self.Um_tilde.T
        # Um_tildeT = self.Um_tilde.T

        rhs = Um_tildeT @ g
        A = Um_tildeT @ H @ Um_tildeT.T 
        term = self.h * self.h * medial_collision_stiffness

        self.b_col_tilde = rhs * term
        self.A_col_tilde = A * term

        self.A0_col = self.Um0.T @ H @ self.Um0 * term
        self.b0_col = self.Um0.T @ g * term

        Um_sys = np.vstack([self.Um0.T, Um_tildeT])
        self.A_sys_col = Um_sys @ H @ Um_sys.T * term
        self.b_sys_col = Um_sys @ g * term
        

    def compute_A(self):
        pass

    def compute_rhs(self):
        pass
        # wp.launch(compute_rhs, (self.n_nodes, ), inputs = [self.states, self.h, self.M, self.b])
        # self.set_bc_fixed_grad()

def staggered_bug():
    
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
    
    # rods = MedialRodComplex(h, meshes, transforms)
    rods = MedialVABD(h, meshes, transforms)
    viewer = MedialViewer(rods)
    ps.set_user_callback(viewer.callback)
    ps.show()


if __name__ == "__main__":
    ps.init()
    ps.look_at((0, 4, 8), (0, 2, 0))
    # ps.set_ground_plane_mode("none")
    # ps.set_ground_plane_height(-collision_eps)
    wp.config.max_unroll = 0
    wp.init()
    # bug_drop()
    staggered_bug()
