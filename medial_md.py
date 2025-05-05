import warp as wp
import numpy as np
import polyscope as ps 
import polyscope.imgui as gui
from geometry.collision_cell import collision_eps
from stretch import h, add_dx, compute_rhs, gravity, gravity_np, PSViewer
from medial_reduced import MedialRodComplex, vec
from mesh_complex import init_transforms, RodComplexBC
from scipy.linalg import lu_factor, lu_solve, solve, polar, inv
from scipy.sparse import *
from scipy.sparse.linalg import splu, norm, spsolve
from g2m.viewer import MedialViewer
from vabd import per_node_forces
from warp.sparse import bsr_zeros, bsr_set_from_triplets, bsr_mv, bsr_axpy
from fem.fem import Triplets
from geometry.static_scene import StaticScene

from g2m.bary_centric import TetBaryCentricCompute
from g2m.medial import SlabMesh
from g2m.collision_medial import MedialCollisionDetector
collider_choice = "medial"
medial_collision_stiffness = 1e7
def vec(t):
    return (t.T).reshape(-1)

class MedialMD(RodComplexBC):
    def __init__(self, h, meshes = [], transforms = [], static_meshes: StaticScene = None):
        model = meshes[0].split("/")[1].split(".")[0]
        self.load_Q(model)
        self.define_z(transforms)
        super().__init__(h, meshes, transforms, static_meshes)

        n_reduced = self.n_reduced
        self.A_reduced = np.zeros((n_reduced, n_reduced))
        self.b_reduced = np.zeros(n_reduced)
        self.define_U()
        self.define_encoder(model)

        self.A_reduced = np.zeros((n_reduced, n_reduced))
        self.b_reduced = np.zeros(n_reduced)
        self.compute_Um()

    def load_Q(self, model):
        Q = np.load(f"data/md/{model}/Q.npy")
        Phi = np.zeros((4, 4, Q.shape[0]))

        for i in range(6, 10):
            for j in range(i, 10):
                Phi[i - 6, j - 6] =  np.load(f"data/md/{model}/Phi_{i}{j}.npy")

        self.Phi = Phi
        self.Q = Q[:, 6: 10]
        
    def define_z(self, transforms):
        n_md = 10
        self.n_modes = self.Q.shape[1] + 12 + n_md
        self.n_meshes = len(transforms)
        self.n_reduced = self.n_modes * self.n_meshes
        self.z = np.zeros((self.n_reduced, ), dtype = float)
        self.dz = np.zeros_like(self.z)
        for i in range(self.n_meshes):
            self.z[i * self.n_modes: i * self.n_modes + 9] = vec(np.identity(3))

    def define_encoder(self, model):
        self.intp = TetBaryCentricCompute(model)

    def get_VR(self):
        V = (self.Um @ self.z).reshape((-1, 3))# + self.V_medial_rest
        # V = self.V_medial_rest
        R = self.R_rest
        return V, R

    def define_U(self):
        self.U = np.zeros((self.n_nodes * 3, self.n_reduced))
        nodes_per_mesh = self.n_nodes // self.n_meshes


        ij_idx = np.array([[i - 6, j - 6] for i in range(6, 10) for j in range(i, 10)])
        self.Phi = np.array([self.Phi[i, j] for i, j in ij_idx]).T
        
        nodes_per_mesh = self.n_nodes // self.n_meshes  
        xnp = self.xcs.numpy()


        U_diags = []
        # self.U = np.zeros((self.n_nodes * 3, self.n_reduced))

        for i in range(self.n_meshes):
            v4 = np.ones((nodes_per_mesh, 4))
            v4[:, :3] = xnp[i * nodes_per_mesh:(i + 1) * nodes_per_mesh]
            v4 = np.kron(v4, np.identity(3))
            uu = np.hstack([v4, self.Q, self.Phi])
            # self.U[i * nodes_per_mesh * 3: (i + 1) * nodes_per_mesh * 3, i * self.n_modes: (i + 1) * self.n_modes] = uu
            U_diags.append(uu)
        self.U = block_diag(U_diags)

    def compute_Um(self):
        nodes_per_mesh = self.n_nodes // self.n_meshes  
        xnp = self.xcs.numpy()
        Um_diags = []
        for i in range(self.n_meshes):
            v4 = np.ones((nodes_per_mesh, 4))
            v4[:, :3] = xnp[i * nodes_per_mesh:(i + 1) * nodes_per_mesh]
            v4 = np.kron(v4, np.identity(3))
            uu = np.hstack([v4, self.Q, self.Phi])
            # self.U[i * nodes_per_mesh * 3: (i + 1) * nodes_per_mesh * 3, i * self.n_modes: (i + 1) * self.n_modes] = uu
            uum = self.intp.compute_deformation(uu)
            Um_diags.append(uum)
        self.Um = block_diag(Um_diags)

    def define_collider(self):
        if collider_choice == "medial":
            self.define_medials()
        else:
            super().define_collider()

    def define_medials(self):
        model = self.meshes_filename[0].split("/")[1].split(".")[0]
        assert model in ["bug", "squishy", "bunny"]
        self.slabmesh = SlabMesh(f"assets/{model}/ma/{model}.ma")
        V0 = np.copy(self.slabmesh.V)
        v4 = np.ones((V0.shape[0], 4))
        v4[:, :3] = V0
        R0 = self.slabmesh.R
        E0 = self.slabmesh.E
        F0 = self.slabmesh.F
        
        R = np.zeros(0, float)
        E = np.zeros((0, 2), int)
        F = np.zeros((0, 3), int)
        V = np.zeros((0, 3))
        self.n_mdeial_per_mesh = V0.shape[0]

        for i in range(self.n_meshes):
            Vi = (v4 @ self.transforms[i].T)[:, : 3]
            cnt = i * self.n_mdeial_per_mesh
            V = np.vstack([V, Vi])
            J3 = np.linalg.det(self.transforms[i][:3, :3])
            J = np.abs(np.power(J3, 1 / 3))
            # R = np.concatenate([R, np.copy(R0) * J])
            R = np.concatenate([R, np.copy(R0)])
            E = np.vstack((E, E0 + cnt))
            F = np.vstack((F, F0 + cnt))

        self.F_medial = F
        self.E_medial = E
        self.V_medial_rest = np.copy(V)
        self.V_medial = np.zeros_like(V)
        self.R_rest = np.copy(R)
        self.R = np.zeros_like(self.R_rest)

        self.V_medial[:] = self.V_medial_rest
        self.R[:] = self.R_rest

        self.collider_medial = MedialCollisionDetector(
            self.V_medial, self.R_rest, self.E_medial, self.F_medial, ground = 0.0, static_objects = self.static_meshes)

        self.n_medial = self.V_medial.shape[0]

    def process_collision(self):
        V, R = self.get_VR()
        self.collider_medial.collision_set(V, R)
        g, H, idx = self.collider_medial.analyze()
        term = self.h * self.h * medial_collision_stiffness

        Ums = self.Um.toarray()[idx].T
        self.b_sys_col = Ums @ (g * term) 
        self.A_sys_col = Ums @ (H * term) @ Ums.T 
        
    def solve(self):
        self.A_reduced = (self.U.T @ self.to_scipy_bsr() @ self.U).toarray()
        b = self.b.numpy().reshape(-1)
        self.b_reduced = self.U.T @ b

        dz = solve(self.A_reduced + self.A_sys_col, self.b_reduced + self.b_sys_col, assume_a = "sym")
        # dz = spsolve(self.A_reduced, self.b_reduced)
        self.dz[:] = dz
        dx = (self.U @ dz).reshape(-1, 3)
        self.states.dx.assign(dx)        
    
    def line_search(self):
        z_tmp = np.copy(self.z)

        alpha = 1.0
        # self.z_tilde[:] = z_tilde_tmp - alpha * self.dz_tilde
        # self.z[:] = z_tmp - alpha * self.dz
        # return alpha

        if True: 
            # zwp = wp.array(self.z.reshape((-1, 3)), dtype = wp.vec3)
            # bsr_mv(self.Uwp, zwp, self.states.x, beta = 0.0)
            x = (self.U @ self.z).reshape(-1, 3)
            self.states.x.assign(x)

        
        e00 = self.compute_psi() + self.compute_inertia()
        e0c = self.compute_collision_energy()
        E0 = e00 + e0c 

        # e01 = self.compute_inertia2()
        # print(f"diff energy = {e01 - e00}, e = {e01}")

        while True:
            self.z[:] = z_tmp - alpha * self.dz

            if True:
                # zwp.assign(self.z.reshape((-1, 3)))
                # bsr_mv(self.Uwp, zwp, self.states.x, beta = 0.0)
                x = (self.U @ self.z).reshape(-1, 3)
                self.states.x.assign(x)

            e10 = self.compute_psi() + self.compute_inertia()
            # e11 = self.compute_inertia2()
            # print(f"diff energy = {e11 - e10}, e = {e11}")

            e1c = self.compute_collision_energy()
            E1 = e10 + e1c
            print(f"e10 = {e10}, e1c = {e1c}, e00 = {e00}, e0c = {e0c}, E1 = {E1}, E0 = {E0}")
            if E1 < E0:
                break
            if alpha < 1e-2:
                self.z[:] = z_tmp
                alpha = 0.0
                break
            alpha *= 0.5
        print(f"line search alpha = {alpha}")
        return alpha
        
    def compute_collision_energy(self):
        V, R = self.get_VR()
        h = self.h
        return self.collider_medial.energy(V, R) * medial_collision_stiffness * h * h

    def reset(self):
        super().reset()
        self.reset_z()
    
    def reset_z(self):
        self.z[:] = 0.0
        self.dz[:] = 0.0
        for i in range(self.n_meshes):
            self.z[i * self.n_modes: i * self.n_modes + 9] = vec(np.identity(3))

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
    rods = MedialMD(h, meshes, transforms, static_bars)
    # viewer = PSViewer(rods, static_bars)
    
    viewer = MedialViewer(rods, static_bars)
    ps.set_user_callback(viewer.callback)
    ps.show()
if __name__ == "__main__":
    ps.init()
    ps.set_ground_plane_height(-collision_eps)
    wp.config.max_unroll = 0
    wp.init()
    staggered_bug()