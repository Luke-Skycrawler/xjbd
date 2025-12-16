import warp as wp 
import numpy as np
from utils.tobj import import_tobj
from scipy.linalg import solve
import igl
import polyscope as ps
from scipy.spatial.transform import Rotation as R
import polyscope.imgui as gui
from modal_warping import ModalWarpingRod, MWViewer
from scipy.io import loadmat

model = "bar2"
save_only = False

class VABDModalWarpingRod(ModalWarpingRod):
    '''
    modal warping rod with VABD basis for fitting modal warping modes
    testing the basis span of different modes including: 
    [ ] fast cd weights 
    [ ] weights from displacment -> PCA
    [ ] weights from (rotation displacement + translation) -> PCA  
    '''
    def __init__(self, filename = f"assets/squishyball/{model}.tobj", add_rotations = False, fast_cd_ref = False):
        super().__init__(filename)
        self.options = {
            "add_rotations": add_rotations,
            "fast_cd_ref": fast_cd_ref
        }
        self.n_weights_limit = 6
        self.load_Q()
        self.define_U()

    def PCA(self, X):
        n_weights = 10
        U, S, V = np.linalg.svd(X)
        Q = np.hstack([np.ones((X.shape[0], 1)), U[:, :n_weights]])
        return Q

    def load_Q(self):

        if not self.options["fast_cd_ref"] and not self.options["add_rotations"]:

            Phi = loadmat(f"data/eigs_lma/Q_{model}.mat")["Vv"].astype(np.float64)[:, 6:]
            Q_norm = np.linalg.norm(Phi, axis = 0, ord = np.inf, keepdims = True)
            Phi /= Q_norm
            Phi = Phi.reshape((Phi.shape[0] // 3, -1))            # Q = np.load(f"data/lma_weight/W_{model}.npy")

            Q = self.PCA(Phi)
            # Q[:, 0] = 1.0
        if self.options["add_rotations"]:
            Psi = np.load(f"data/lma_weight/Psi_{model}.npy")[:, 6:]
            Q_norm = np.linalg.norm(Psi, axis = 0, ord = np.inf, keepdims = True)
            Psi /= Q_norm
            Psi = Psi.reshape((-1, 3 * Psi.shape[1]))


            Phi = loadmat(f"data/eigs_lma/Q_{model}.mat")["Vv"].astype(np.float64)[:, 6:]
            Q_norm = np.linalg.norm(Phi, axis = 0, ord = np.inf, keepdims = True)
            Phi /= Q_norm
            Phi = Phi.reshape((Phi.shape[0] // 3, -1))


            Q = self.PCA(np.hstack([Psi, Phi]))
            if save_only:
                arr = []
                assert Psi.shape[1] == Phi.shape[1]
                for i in range(Psi.shape[1]):
                    ps = Psi[:, i: i + 1]
                    ph = Phi[:, i: i + 1]
                    arr.append(ps)
                    arr.append(ph)
                data = np.hstack(arr)
                np.save(f"data/lma_weight/all_weights_{model}.npy", data)
                quit()
            # Q = self.PCA(np.hstack([Phi]))

        if self.options["fast_cd_ref"]:
            Q = np.load(f"data/W_{model}.npy")
            Q[:, 0] = 1.0
        self.Q_vabd = Q[:, :self.n_weights_limit]
        
    def define_U(self):
        x0 = self.xcs.numpy()
        self.U = self.lbs_matrix(x0, self.Q_vabd)
        self.z = np.zeros((self.U.shape[1], ))

    def lbs_matrix(self, V, W):
        nvm = V.shape[0]
        v1 = np.ones((nvm, 4))
        v1[:, :3] = V
        lhs = np.hstack([W[:, j: j + 1] * v1 for j in range(W.shape[1])])
        return np.kron(lhs, np.identity(3))

    def reconstruct(self):
        return (self.U @ self.z).reshape((-1, 3))

    def best_fit_transform(self, disp):
        U = self.U
        A = U.T @ U
        b = U.T @ disp.reshape(-1)
        self.z[:] = solve(A, b, assume_a="sym")

class BestFitViewer(MWViewer):
    def __init__(self, rod: VABDModalWarpingRod):
        super().__init__(rod)
        self.disp = np.zeros_like(self.V0)
        self.show_best_fit = False
        self.ui_basis_type = 0  # 0: fast cd, 1: translation, 2: translation + rotation

    def control_panel(self):
        super().control_panel()
        if gui.Button("best fit tansform"):
            self.rod.best_fit_transform(self.disp)
        changed, self.show_best_fit = gui.Checkbox("Show Best Fit", self.show_best_fit)
        changed, self.rod.n_weights_limit = gui.InputInt("n_weights", self.rod.n_weights_limit, step = 1)
        changed2, self.ui_basis_type = gui.SliderInt("Basis Type", self.ui_basis_type, v_min = 0, v_max=2)

        if changed2: 
            self.rod.options["add_rotations"] = (self.ui_basis_type == 2)
            self.rod.options["fast_cd_ref"] = (self.ui_basis_type == 0)

        if changed or changed2: 
            self.rod.load_Q()
            self.rod.define_U()
            self.rod.best_fit_transform(self.disp)
        
    
    def display(self):
        # self.disp = self.rod.compute_displacement(self.ui_deformed_mode, self.ui_magnitude) if self.ui_use_modal_warping else (self.Q[:, self.ui_deformed_mode] * self.ui_magnitude).reshape((-1, 3))
        
        # self.disp = self.rod.compute_displacement_mix(self.ui_deformed_mode, self.ui_magnitude)
        blend = np.dot(self.Q, self.qs)
        self.disp = self.rod.compute_displacement_mix(blend)

        if not self.show_best_fit:
            self.V_deform = self.V0 + self.disp 
        else: 
            self.V_deform = self.V0 + self.rod.reconstruct()


# def compute_ortho_loss():
#     Q = np.load(f"data/W_{model}.npy")
#     Q[:, 0] = 1.0

#     x_target = np.load(f"data/x_target.npy")
    
#     # r = R.from_rotvec([0, 0, np.pi/4])  # 45° around Z
#     # rot_matrix = r.as_matrix()

#     # x_target = x_target @ rot_matrix.T
#     t = np.mean(x_target, axis = 0, keepdims= True)

#     v, T = import_tobj(f"assets/squishyball/{model}.tobj")
#     F = igl.boundary_facets(T)
#     u = x_target - v

#     U = lbs_matrix(v, Q[:, :])
    
#     A = U.T @ U
#     b = U.T @ u.reshape(-1)
#     z = solve(A, b, assume_a="sym")
#     x = (U @ z).reshape((-1, 3)) + v

#     translated = v + t

#     viewer = PSViewer(x, v, F, x_target, U, z)
    

if __name__ == "__main__":
    wp.init()
    ps.init()
    ps.set_ground_plane_mode("none")

    # rod = VABDModalWarpingRod(f"assets/{model}/{model}.tobj", add_rotations=True)
    rod = VABDModalWarpingRod(f"assets/{model}/{model}.tobj", add_rotations=True)
    # rod = VABDModalWarpingRod(f"assets/{model}/{model}.tobj", fast_cd_ref=True)

    viewer = BestFitViewer(rod)
    ps.set_user_callback(viewer.callback)

    ps.show()