import warp as wp
import numpy as np
import polyscope as ps 
import polyscope.imgui as gui
from geometry.collision_cell import collision_eps
from stretch import h, add_dx, compute_rhs, gravity, gravity_np, PSViewer
from medial_reduced import MedialRodComplex, vec
from mesh_complex import init_transforms
from scipy.linalg import lu_factor, lu_solve, solve, polar, inv
from scipy.sparse import *
from scipy.sparse.linalg import splu, norm
from ortho import OrthogonalEnergy
from g2m.viewer import MedialViewer
from vabd import per_node_forces
from warp.sparse import bsr_zeros, bsr_set_from_triplets, bsr_mv, bsr_axpy
from fem.fem import Triplets
from fem.interface import TOBJComplex, StaticScene
ad_hoc = True
medial_collision_stiffness = 1e4
solver_choice = "woodbury"
assert solver_choice in ["woodbury", "direct", "compare"]
def asym(a):
    return 0.5 * (a - a.T)

@wp.kernel
def fill_U_triplets(mesh_id: int, xcs: wp.array(dtype = wp.vec3), W: wp.array2d(dtype = float), triplets: Triplets):
    i, j, k = wp.tid()
    xx = W.shape[0]
    yy = W.shape[1]
    block_nnz = 4 * xx * yy

    idx = (i * yy + j) * 4 + k + block_nnz * mesh_id
    xid = i + mesh_id * xx
    triplets.rows[idx] = xid
    triplets.cols[idx] = j * 4 + k + mesh_id * yy * 4
    c = float(1.0)
    if k < 3:
        c = xcs[xid][k]
    triplets.vals[idx] = wp.diag(wp.vec3(W[i, j] * c))


class WoodburySolver:
    def __init__(self, A0_dim, A_tilde):
        self.A_tilde = A_tilde
        # self.lu, self.piv = lu_factor(self.A_tilde)
        self.tilde_solve = splu(self.A_tilde)
        self.A0 = np.zeros((A0_dim, A0_dim))
        self.A0_dim = A0_dim
        self.k = 0
    
    def update(self, A0, U, C, V):
        self.A0[:] = A0
        self.U = U.toarray()
        self.C = C
        self.V = V.toarray()

        self.lu0, self.piv0 = lu_factor(self.A0)
        
        self.k = self.U.shape[1]
        self.add_rank = self.k > 0 and np.linalg.det(self.C) > 1e-5
        if self.add_rank:
            self.A_inv_U = self.apply_inv_A(self.U)
            self.central_term = inv(self.C) + self.V @ self.A_inv_U 

    def apply_inv_A(self, b):
        # x_tilde = lu_solve((self.lu, self.piv), b[self.A0_dim:])
        x_tilde = self.tilde_solve.solve(b[self.A0_dim:])

        x0 = lu_solve((self.lu0, self.piv0), b[:self.A0_dim])
        # x0 = solve(self.A0, b[:self.A0_dim])
        return np.concatenate([x0, x_tilde])
    
    def solve(self, b):
        vi = self.apply_inv_A(b) 
        term1 = vi

        if self.add_rank:
            VA_inv = self.V @ vi 
            
            tmp = solve(self.central_term, VA_inv)
            term2 = self.A_inv_U @ tmp
        else:
            return term1

        return term1 - term2 

class MedialVABD(MedialRodComplex):
    def __init__(self, h, meshes=[], transforms=[]):
        super().__init__(h, meshes, transforms)
        self.split_U0_U_tilide()
        self.define_mm()
        self.prefactor_once()
        self.define_A0_b0_btilde()
        
        self.ortho = OrthogonalEnergy(self.n_meshes)
        self.sum_weights()

        self.gen_F_idx()
        g = np.zeros(12)
        g[9:12] = gravity_np
        self.gravity = np.concatenate([g for _ in range(self.n_meshes)])

    def gen_F_idx(self):
        '''
        optioal, only used in optimizing fetching the deformation gradient 
        '''
        idx = [np.arange(i * self.n_modes, i * self.n_modes + 9) for i in range(self.n_meshes)]
        self.F_idx = np.array(idx)

    def sum_weights(self):
        ww  = self.W.numpy()
        n_nodes_per_mesh = self.n_nodes // self.n_meshes
        weights = ww.reshape((self.n_meshes, -1))
        self.sum_W = np.sum(weights, axis = 1, keepdims = False)

    def define_A0_b0_btilde(self):
        self.A0 = np.zeros((self.n_meshes * 12, self.n_meshes * 12))
        self.b0 = np.zeros((self.n_meshes * 12,))
        self.b_tilde = np.zeros((self.n_meshes * (self.n_modes - 12),))


    def define_U(self):
        # self.U = np.zeros((self.n_nodes * 3, self.n_reduced))
        nodes_per_mesh = self.n_nodes // self.n_meshes
        x0 = self.xcs.numpy()
        diags = []
        
        for i in range(self.n_meshes): 
            xi = x0[i * nodes_per_mesh: (i + 1) * nodes_per_mesh]
            Ui = self.lbs_matrix(xi, self.Q)
            diags.append(Ui)
        self.U = block_diag(diags)

        self.Uwp = bsr_zeros(self.n_nodes, self.n_modes // 3 * self.n_meshes, wp.mat33)
        q = wp.array(self.Q, dtype = float)
        triplets = Triplets()
        nnz = q.shape[0] * q.shape[1] * 4 * self.n_meshes
        triplets.cols = wp.zeros((nnz, ), int)
        triplets.rows = wp.zeros((nnz, ), int)
        triplets.vals = wp.zeros((nnz,), wp.mat33)

        for i in range(self.n_meshes):
            wp.launch(fill_U_triplets, (q.shape[0], q.shape[1], 4), inputs = [i, self.geo.xcs, q, triplets])
        bsr_set_from_triplets(self.Uwp, triplets.rows, triplets.cols, triplets.vals, )
        
        # Uwp = self.to_scipy_bsr(self.Uwp)
        # print(f"U norm = {norm(self.U)}, diff norm = {norm(self.U - Uwp)}")

    def split_U0_U_tilide(self):
        # self.U0 = np.zeros((self.n_nodes * 3, self.n_meshes * 12))
        # self.U_tilde = np.zeros((self.n_nodes * 3, (self.n_modes - 12) * self.n_meshes))
        n_nodes_per_mesh = self.n_nodes // self.n_meshes
        x0np = self.xcs.numpy()
        diags0 = []
        diags_tilde = []
        for i in range(self.n_meshes):
            start = i * n_nodes_per_mesh * 3
            end = start + n_nodes_per_mesh * 3
            
            V = x0np[i * n_nodes_per_mesh: (i + 1) * n_nodes_per_mesh]
            w = self.Q
            jac = self.lbs_matrix(V, w)
            
            diags0.append(jac[:, :12])
            diags_tilde.append(jac[:, 12:])
            # self.U0[start: end, 12 * i: 12 * (i + 1)] = rhs


            # self.U_tilde[start:end, i * (self.n_modes - 12): (i + 1) * (self.n_modes - 12)] = self.lbs_matrix(V, self.Q[:, 1:])

        self.U0 = block_diag(diags0)
        self.U_tilde = block_diag(diags_tilde)

        # fill Um_tilde 
        
        # self.Um_tilde = np.zeros((self.n_medial * 3, self.z_tilde.shape[0]))
        # self.Um0 = np.zeros((self.n_medial * 3, 12 * self.n_meshes))
        diags0 = []
        diags_tilde = []
        self.diags_tilde_lhs = []
        for i in range(self.n_meshes):
            # start = i * self.n_modes
            # end = (i + 1) * self.n_modes

            # self.Um_tilde[:, i * (self.n_modes - 12): (i + 1) * (self.n_modes - 12)] = self.Um[:, start + 12: end]
            # self.Um0[:, i * 12: (i + 1) * 12] = self.Um[:, start: start + 12]
            Vmi = self.V_medial_rest[i * self.n_mdeial_per_mesh: (i + 1) * self.n_mdeial_per_mesh]

            ji = self.lbs_matrix(Vmi, self.W_medial)
            j0i = ji[:,: 12]
            jti = ji[:, 12:]
            diags0.append(j0i)
            diags_tilde.append(jti)

            di = self.lbs_no_kron(Vmi, self.W_medial[:, 1:])
            self.diags_tilde_lhs.append(di)

        self.Um0 = block_diag(diags0)
        self.Um_tilde = block_diag(diags_tilde)

    def define_mm(self):
        self.mm = self.U0.T @ self.to_scipy_bsr(self.M_sparse) @ self.U0
    
    def lbs_no_kron(self, V, W):
        nvm = V.shape[0]
        v1 = np.ones((nvm, 4))
        v1[:, :3] = V
        lhs = np.hstack([W[:, j: j + 1] * v1 for j in range(W.shape[1])])
        return lhs

    def get_F(self, i):
        '''
        columns of F is formed by segments of z 
        vec(F) = z
        '''
        z = self.z[i * self.n_modes: i * self.n_modes + 12]
        F = z.reshape((-1, 3)).T
        return F[:, :3]

    def get_F_batch(self):
        '''
        columns of F is formed by segments of z 
        vec(F) = z
        '''
        f = self.z[self.F_idx].reshape((self.n_meshes, 3, 3)) # n_meshes x 3 x3
        return np.transpose(f, axes = (0, 2, 1))

    def prefactor_once(self):
        h = self.h
        self.compute_K()
        self.K0 = self.U_tilde.T @ self.to_scipy_bsr() @ self.U_tilde * (h * h)
        self.M_tilde = self.U_tilde.T @ self.to_scipy_bsr(self.M_sparse) @ self.U_tilde

        self.A_tilde = self.K0 + self.M_tilde

        # cholesky factorization tend to have non-positive definite matrix, use lu instead
        # self.c, self.low = lu_factor(self.A_tilde)
        self.solver = WoodburySolver(self.n_meshes * 12, self.A_tilde)

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

    def z_tilde2z(self, z, ztilde):
        for i in range(self.n_meshes):
            fi = self.get_F(i)
            R, _ = polar(fi)
            start = i * (self.n_modes) + 12
            end = (i + 1) * self.n_modes
            
            zz= R @ (ztilde[i * (self.n_modes -12) : (i+ 1) * (self.n_modes - 12)].reshape((-1, 3)).T)
            self.z[start:end] = (zz.T).reshape(-1)

    def update_x0_xdot(self):
        # super().update_x0_xdot()

        z = self.extract_z0(self.z)
        self.z_tilde2z(z, self.z_tilde)

        self.z_dot[:] = (z - self.z0) / self.h
        self.z0[:] = z

        self.z_tilde_dot[:] = (self.z_tilde - self.z_tilde0) / self.h
        self.z_tilde0[:] = self.z_tilde

        # self.states.x.assign((self.U @ self.z).reshape((-1, 3)))
        zwp = wp.array(self.z.reshape((-1, 3)), dtype = wp.vec3)
        bsr_mv(self.Uwp, zwp, self.states.x, beta = 0.0)

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
        with wp.ScopedTimer("compute A0"):
            Fs = [self.get_F(i) for i in range(self.n_meshes)]
            gg, HH = self.ortho.analyze(np.array(Fs))
            for i in range(self.n_meshes):
                # fi = self.get_F(i)

                # g, H = self.ortho.analyze(fi)
                g = gg[i]
                H = HH[i]
                mmi = self.mm[i * 12: (i + 1) * 12, i * 12: (i + 1) * 12]
                self.A0[i * 12: (i + 1) * 12, i * 12: (i + 1) * 12] = H * h2 * self.sum_W[i] + mmi
                self.b0[i * 12: (i + 1) * 12] = g * h2 * self.sum_W[i] + mmi @ (self.z[i * self.n_modes: i * self.n_modes + 12] - self.z_hat(i))

        # set b_tilde
        self.b_tilde = self.K0 @ self.z_tilde + self.M_tilde @ (self.z_tilde - self.z_tilde_hat()) + self.compute_excitement()

        # dz0 = solve(self.A0, self.b0, assume_a="sym")
        # dz0 = solve(self.A0 + self.A0_col, self.b0 + self.b0_col, assume_a="sym")

        
        # self.dz_tilde = lu_solve((self.c, self.low), self.b_tilde)

        if solver_choice in ["direct", "compare"]:
            
            A_sys = np.zeros((self.n_reduced, self.n_reduced))
            


            # U_prime = self.compute_U_prime()
            # U_tildeT = U_prime @ self.U_tilde.T
            # U_sys = np.vstack([self.U0.T, U_tildeT])

            # A_sys = U_sys @ self.to_scipy_bsr() @ U_sys.T 
            # b_sys = U_sys @ self.b.numpy().reshape(-1) 

            A_sys[:self.n_meshes * 12, :self.n_meshes * 12] = self.A0# + self.A0_col
            # A_sys[:self.n_meshes * 12, self.n_meshes * 12:] = 0.0
            # A_sys[self.n_meshes * 12:, :self.n_meshes * 12] = 0.0
            # np.save("A_tilde.npy", A_sys[self.n_meshes * 12:, self.n_meshes * 12:])
            # np.save("A_tilde_K0.npy", self.A_tilde)
            A_sys[self.n_meshes * 12:, self.n_meshes * 12:] = self.A_tilde.toarray()# + self.A_col_tilde

        b_sys = np.zeros(self.n_reduced)
        b_sys[:self.n_meshes * 12] = self.b0# + self.b0_col
        b_sys[self.n_meshes * 12:] = self.b_tilde# + self.b_col_tilde

        if solver_choice in ["direct", "compare"]:
            with wp.ScopedTimer("linalg system"): 
                dz_sys = solve(A_sys + self.A_sys_col, b_sys + self.b_sys_col, assume_a="sym")

        if solver_choice in ["woodbury", "compare"]:
            with wp.ScopedTimer("woodbury solver"): 
                self.solver.update(self.A0, self.U_col, self.C_col, self.U_col.T)
                dz_sys_wb = self.solver.solve(b_sys + self.b_sys_col)
        
        if solver_choice == "compare":
            print(f"diff from solvers = {np.linalg.norm(dz_sys - dz_sys_wb)}")
        elif solver_choice == "woodbury":
            dz_sys = dz_sys_wb

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

    def converged(self):
        norm_dz = np.linalg.norm(self.dz)
        return norm_dz < 1e-3
        
    def line_search(self):
        z_tilde_tmp = np.copy(self.z_tilde)
        z_tmp = np.copy(self.z)

        alpha = 1.0

        zwp = wp.array(self.z.reshape((-1, 3)), dtype = wp.vec3)
        bsr_mv(self.Uwp, zwp, self.states.x, beta = 0.0)
        
        e00 = self.compute_psi() + self.compute_inertia()
        E0 = e00 + self.compute_collision_energy()

        # e01 = self.compute_inertia2()
        # print(f"diff energy = {e01 - e00}, e = {e01}")

        while True:
            self.z_tilde[:] = z_tilde_tmp - alpha * self.dz_tilde
            self.z[:] = z_tmp - alpha * self.dz

            zwp.assign(self.z.reshape((-1, 3)))
            bsr_mv(self.Uwp, zwp, self.states.x, beta = 0.0)

            e10 = self.compute_psi() + self.compute_inertia()
            # e11 = self.compute_inertia2()
            # print(f"diff energy = {e11 - e10}, e = {e11}")

            E1 = e10 + self.compute_collision_energy()
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
        with wp.ScopedTimer("Psi"):
            h2 = self.h ** 2
            ff = self.get_F_batch()
            e = self.ortho.energy_batch(ff)
            ret = np.sum(e * self.sum_W) * h2
        return ret + self.z_tilde @ self.K0 @ self.z_tilde * 0.5

    def compute_inertia(self):
        with wp.ScopedTimer("inertia"):

            zh = self.z0 + self.h * self.z_dot + self.gravity * self.h * self.h
            z0 = self.extract_z0(self.z)
            dz0 = z0 - zh

            dzt = self.z_tilde - self.z_tilde_hat()
            ret = (dz0 @ self.mm @ dz0 + dzt @ self.M_tilde @ dzt) * 0.5
        return ret
    
    # def compute_inertia2(self):
    #     h2 = self.h ** 2
    #     ret = 0.0
    #     for i in range(self.n_meshes):
    #         fi = self.get_F(i)
    #         e = self.ortho.energy(fi)
    #         mmi = self.mm[i * 12: (i + 1) * 12, i * 12: (i + 1) * 12]

    #         dz = self.z[i * self.n_modes: i * self.n_modes + 12] - self.z_hat(i)
    #         ret += e * h2 * self.sum_W[i] + 0.5 * dz @ mmi @ dz

    #     dzt = self.z_tilde - self.z_tilde_hat()
    #     ret += (self.z_tilde @ self.K0 @ self.z_tilde + dzt @ self.M_tilde @ dzt) * 0.5
    #     return ret

    def compute_collision_energy(self):
        # self.n_pt, self.n_ee, self.n_ground = self.collider.collision_set("all")
        # return self.collider.collision_energy(self.n_pt, self.n_ee, self.n_ground) * self.h * self.h
        with wp.ScopedTimer("get V, R"):
            V, R = self.get_VR()
        h = self.h
        with wp.ScopedTimer("energy"):
            ret = self.collider_medial.energy(V, R) * medial_collision_stiffness * h * h    
        return ret

    # def line_search(self):
    #     self.z_tilde -= self.dz_tilde
    #     self.z -= self.dz
        
    #     # self.states.x.assign((self.U @ self.z).reshape((-1, 3)))
    #     return 1.0
        
    def z_tilde_hat(self):
        return self.h * self.z_tilde_dot + self.z_tilde0

    def z_hat(self, i):
        zh = self.z0 + self.h * self.z_dot
        ret = zh[i * 12: (i + 1) * 12]
        ret[9: 12] += gravity_np * self.h * self.h
        return ret

    def compute_U_prime(self):
        U_prime_dim = self.z_tilde.shape[0]
        # U_prime = np.zeros((U_prime_dim, U_prime_dim))
        diags = []
        for i in range(self.n_meshes):
            fi = self.get_F(i)
            ri, _ = polar(fi)
            
            i_m = np.identity((self.n_modes - 12) // 3, float)
            start  = i * (self.n_modes - 12)
            end = start + (self.n_modes - 12)
            # U_prime[start:end, start: end] = np.kron(i_m, ri)
            m = (self.n_modes - 12) // 3
            diags += [ri] * m
        U_prime = block_diag(diags, )
        assert U_prime.shape[0] == U_prime_dim
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
        
        if self.n_meshes == 2: 
            self.z_dot[12 + 9: 12 + 12] = np.array([0.0, 0.0, -3.0])
            
            # self.z_dot[2] = -1.0
            # self.z_dot[6] = 1.0
        else:
            for i in range(self.n_meshes):
                ti = self.transforms[i][:3, 3]
                self.z_dot[i * 12 + 9: i * 12 + 12] = -ti * 0.25

        self.z_tilde[:] = 0.0
        self.z_tilde0[:] = 0.0
        self.z_tilde_dot[:] = 0.0
        self.dz_tilde[:] = 0.0

        self.frame = 0

    def compute_Um_tildeT(self):
        diags = []
        for i in range(self.n_meshes):
            fi = self.get_F(i)
            # R, _ = polar(fi)
            R = fi
            diags.append(np.kron(self.diags_tilde_lhs[i].T, R.T))
        with wp.ScopedTimer("build csc"):
            ret = block_diag(diags, "csc")
        return ret

    def process_collision(self):
        V, R = self.get_VR()
        with wp.ScopedTimer("detect"):
            self.collider_medial.collision_set(V, R)
        with wp.ScopedTimer("analyze"):
            g, H, idx = self.collider_medial.analyze()
            
        # with wp.ScopedTimer("U prime"):
        #     U_prime = self.compute_U_prime()
        
        # with wp.ScopedTimer("prod 1"):
        #     Um_tildeT = U_prime @ self.Um_tilde.T

        with wp.ScopedTimer("Um tildeT"):
            Um_tildeT = self.compute_Um_tildeT()

        # print(f"diff Um = {norm(Um_tildeT - Um_tildeT1)}, Um norm = {norm(Um_tildeT)}")

        term = self.h * self.h * medial_collision_stiffness
        with wp.ScopedTimer("collision linalg system"):
            # rhs = Um_tildeT @ g
            # A = Um_tildeT @ H @ Um_tildeT.T 

            # self.b_col_tilde = rhs * term
            # self.A_col_tilde = A * term

            # self.A0_col = self.Um0.T @ H @ self.Um0 * term
            # self.b0_col = self.Um0.T @ g * term

            Um_sys = vstack([self.Um0.T, Um_tildeT]).tocsc()[:, idx]
            # Um_sys = np.vstack([self.Um0.T, Um_tildeT])[:, idx]
            step1 = Um_sys @ (H * term)
            # self.A_sys_col = Um_sys @ (H * term) @ Um_sys.T 
            self.b_sys_col = Um_sys @ (g * term)

            self.A_sys_col = step1 @ Um_sys.T
        
        self.U_col = Um_sys
        self.C_col = H

    # def process_collision(self):
    #     self.b.zero_()
    #     with wp.ScopedTimer("collision"):
    #         with wp.ScopedTimer("detection"):
    #             self.n_pt, self.n_ee, self.n_ground = self.collider.collision_set("all") 
    #         with wp.ScopedTimer("hess & grad"):
    #             triplets = self.collider.analyze(self.b, self.n_pt, self.n_ee, self.n_ground)
    #             # triplets = self.collider.analyze(self.b)
    #         with wp.ScopedTimer("build_from_triplets"):

    #             collision_force_derivatives = bsr_zeros(self.n_nodes, self.n_nodes, wp.mat33)
    #             bsr_set_from_triplets(collision_force_derivatives, triplets.rows, triplets.cols, triplets.vals)
    #             # bsr_axpy(collision_force_derivatives, self.A, self.h * self.h)
    #             H = self.to_scipy_bsr(collision_force_derivatives)
    #             g = self.b.numpy().reshape(-1)


    #             # self.add_collision_to_sys_matrix(triplets)

    #             U_prime = self.compute_U_prime()
        
    #             U_tildeT = U_prime @ self.U_tilde.T
    #             U_sys = vstack([self.U0.T, U_tildeT])
    #             self.A_sys_col = U_sys @ (H * h * h) @ U_sys.T
    #             self.b_sys_col = U_sys @ -g * h * h


    #     with wp.ScopedTimer("Um tildeT"):
    #         Um_tildeT = self.compute_Um_tildeT()

    #     # print(f"diff Um = {norm(Um_tildeT - Um_tildeT1)}, Um norm = {norm(Um_tildeT)}")

    #     term = self.h * self.h * medial_collision_stiffness
    #     with wp.ScopedTimer("collision linalg system"):
    #         # rhs = Um_tildeT @ g
    #         # A = Um_tildeT @ H @ Um_tildeT.T 

    #         # self.b_col_tilde = rhs * term
    #         # self.A_col_tilde = A * term

    #         # self.A0_col = self.Um0.T @ H @ self.Um0 * term
    #         # self.b0_col = self.Um0.T @ g * term

    #         Um_sys = vstack([self.Um0.T, Um_tildeT]).tocsc()[:, idx]
    #         # Um_sys = np.vstack([self.Um0.T, Um_tildeT])[:, idx]
    #         step1 = Um_sys @ (H * term)
    #         # self.A_sys_col = Um_sys @ (H * term) @ Um_sys.T 
    #         self.b_sys_col = Um_sys @ (g * term)

    #         self.A_sys_col = step1 @ Um_sys.T

    def compute_A(self):
        pass

    def compute_rhs(self):
        pass
        # wp.launch(compute_rhs, (self.n_nodes, ), inputs = [self.states, self.h, self.M, self.b])
        # self.set_bc_fixed_grad()

# def staggered_bug():
    
#     n_meshes = 2 
#     meshes = ["assets/bug.tobj"] * n_meshes
#     # meshes = ["assets/bunny_5.tobj"] * n_meshes
#     transforms = [np.identity(4, dtype = float) for _ in range(n_meshes)]
#     transforms[1][:3, :3] = np.zeros((3, 3))
#     transforms[1][0, 1] = 1
#     transforms[1][1, 0] = 1
#     transforms[1][2, 2] = 1

#     for i in range(n_meshes):
#         # transforms[i][0, 3] = i * 0.5
#         transforms[i][1, 3] = 1.2 + i * 0.2
#         transforms[i][2, 3] = i * 1.0
    
#     # rods = MedialRodComplex(h, meshes, transforms)
#     rods = MedialVABD(h, meshes, transforms)
#     viewer = MedialViewer(rods)
#     ps.set_user_callback(viewer.callback)
#     ps.show()

def staggered_bug():
    
    n_meshes = 2 
    # meshes = ["assets/bug.tobj"] * n_meshes
    meshes = ["assets/squishyball/squishy_ball_lowlow.tobj"] * n_meshes
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
    static_meshes_file = ["assets/teapotContainer.obj"]
    scale = np.identity(4) * 3
    scale[3, 3] = 1.0
    static_bars = StaticScene(static_meshes_file, np.array([scale]))
    # static_bars = None
    rods = MedialVABD(h, meshes, transforms)
    
    
    viewer = MedialViewer(rods)
    ps.set_user_callback(viewer.callback)
    ps.show()

def bug_rain():
    n_meshes = 20
    meshes = ["assets/bug.tobj"] * n_meshes
    
    transforms = wp.zeros((n_meshes, ), dtype = wp.mat44)
    # v, _ = import_tobj(meshes[0])
    # bb_size = np.max(v, axis = 0) - np.min(v, axis = 0)
    bb_size = np.ones(3, float)
    wp.launch(init_transforms, (n_meshes,), inputs = [transforms, bb_size[0], bb_size[1], bb_size[2]])
    print(f"bb_size = {bb_size}")
    rods = MedialVABD(h, meshes, transforms.numpy())
    viewer = MedialViewer(rods)
    ps.set_user_callback(viewer.callback)
    ps.show()

if __name__ == "__main__":
    ps.init()
    ps.look_at((0, 4, 8), (0, 2, 0))
    # ps.set_ground_plane_mode("none")
    ps.set_ground_plane_height(-collision_eps)
    wp.config.max_unroll = 0
    wp.init()
    staggered_bug()
    # bug_rain()
