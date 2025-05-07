import numpy as np
import igl
from sqem import Sqem
class SQEMFitter:
    def __init__(self, V, F, V_medial, R):
        self.V = V
        self.V0 = np.copy(V)
        self.F = F
        self.V_medial_rest = np.copy(V_medial)
        self.V_medial = np.copy(V_medial)
        self.R = np.copy(R)
        self.R_rest = np.copy(R)

        self.nv_surface =V.shape[0]
        self.nv_medial = V_medial.shape[0]

        self.VF, self.NI = igl.vertex_triangle_adjacency(F, self.nv_surface)
        
        self.nearest = np.zeros((self.nv_surface), dtype = np.int32)
        self.fitted_V = np.copy(V_medial)
        self.fitted_R = np.copy(R)
        
        self.assign_nearest()

    def assign_nearest(self):
        
        for i in range(self.nv_surface):

            v = self.V0[i] 
            
            d = np.abs(np.linalg.norm(self.V_medial_rest - v, axis = 1) - self.R)
            
            self.nearest[i] = np.argmin(d)

    def fit_spheres(self, V_medial, R):
        threshold = 5
        self.N = igl.per_face_normals(self.V, self.F, np.array([0.0, 0.0, 1.0], dtype = np.float32))

        for i in range(V_medial.shape[0]):
            select = self.nearest == i
            n_select = np.arange(self.nv_surface)[select]

            if len(n_select) < threshold:
                self.fitted_V[i] = V_medial[i]
                self.fitted_R[i] = R[i]

            center, r = self.sphere_from_patch(select)
            # slab_V[i] = center
            # slab_R[i] = r

            self.fitted_V[i] = center
            self.fitted_R[i] = r
            
        return self.fitted_V, self.fitted_R

    def sphere_from_patch(self, select):
        faces_set = self.face_set_from_points(select)

        
        sum = Sqem()
        sum.set_zero()
        for f in faces_set:
            p = self.V[self.F[f, 0]]
            n = self.N[f]
            sf = Sqem(p, n)
            sum = sum + sf

        pa = np.ones((3)) * -10.0
        pb = np.ones((3)) * 10.0
        center,  r = sum.minimize(pa, pb)
        return center, r

    def face_set_from_points(self, bool_selection):
        faces_set = np.zeros((0), dtype = np.int32)
        idx = np.arange(self.nv_surface)[bool_selection]

        for v in idx:
            fv = self.VF[self.NI[v]: self.NI[v + 1]]
            faces_set = np.concatenate((faces_set, fv))

        return faces_set
