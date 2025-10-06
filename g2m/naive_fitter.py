import numpy as np
import igl
from sqem import Sqem
from scipy.spatial.transform import Rotation as R

dqs_enabled = True
class Fitter: 
    def __init__(self, V, T, slabmesh):
        self.surface_F = igl.boundary_facets(T)
        self.nv_surface = np.max(self.surface_F) + 1
        self.V = V
        self.slabmesh = slabmesh
        self.surface_V = self.V[:self.nv_surface]
        
        self.nv_surface = self.surface_V.shape[0]
        self.nv_medial = slabmesh.V.shape[0]

        self.V_deform = np.copy(V)
        self.VF, self.NI = igl.vertex_triangle_adjacency(self.surface_F, self.surface_V.shape[0])

        self.nearest = np.zeros((self.nv_surface), dtype = np.int32)
        

        self.fitted_V = slabmesh.V.copy()
        self.fitted_R = slabmesh.R.copy()

        self.assign_nearest()

    def assign_nearest(self):
        
        for i in range(self.nv_surface):

            # TODO: check if self.surface_V is better 
            v = self.V_deform[i] 

            d = np.abs(np.linalg.norm(self.slabmesh.V - v, axis = 1) - self.slabmesh.R)
            self.nearest[i] = np.argmin(d)

    def fit_spheres(self):

        threshold = 5
        self.N = -igl.per_face_normals(self.V_deform, self.surface_F, np.array([0.0, 0.0, 1.0]))
        # FIXME: sometimes need to flip the normals to get a positive r, or need to explicitly orient the normals outward

        slab_V = self.slabmesh.V
        slab_R = self.slabmesh.R
        for i in range(slab_V.shape[0]):
            select = self.nearest == i
            n_select = np.arange(self.nv_surface)[select]

            if len(n_select) < threshold:
                self.fitted_V[i] = slab_V[i]
                self.fitted_R[i] = slab_R[i]
            else:
                center, r = self.sphere_from_patch(select)

                self.fitted_V[i] = center
                self.fitted_R[i] = r
            
    def sphere_from_patch(self, select):
        faces_set = self.face_set_from_points(select)

        
        sum = Sqem()
        sum.set_zero()
        for f in faces_set:
            p = self.V_deform[self.surface_F[f, 0]]
            n = self.N[f]
            sf = Sqem(p, n)
            sum = sum + sf

        pa = np.ones((3)) * -10.0
        pb = np.ones((3)) * 10.0
        center,  r = sum.minimize(pa, pb)
        return center, r
              
    def V2p(self, V):
        self.V_deform[:] = V
        self.fit_spheres() 
        return self.fitted_V, self.fitted_R

    def face_set_from_points(self, bool_selection):
        faces_set = np.zeros((0), dtype = np.int32)
        idx = np.arange(self.nv_surface)[bool_selection]

        for v in idx:
            fv = self.VF[self.NI[v]: self.NI[v + 1]]
            faces_set = np.concatenate((faces_set, fv))

        return faces_set

