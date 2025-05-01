from fem.interface import TOBJComplex
from g2m.medial import SlabMesh
import os
import numpy as np
class StaticScene(TOBJComplex):
    def __init__(self, meshes = [], transforms = []):
        self.meshes_filename = meshes
        self.transforms = transforms
        super().__init__()
        self.define_medials()
        
            
    def define_medials(self):
        assert len(self.meshes_filename) == 1
        f = self.meshes_filename[0]
        f_ext = f.split(".")[-1]
        f_medial = f.replace(f_ext, "ma")
        
        if os.path.exists(f_medial):
            slabmesh = SlabMesh(f_medial)

            V0 = np.copy(slabmesh.V)
            v4 = np.ones((V0.shape[0], 4))
            v4[:, :3] = V0
            R0 = slabmesh.R
            E0 = slabmesh.E
            
            R = np.zeros(0, float)
            E = np.zeros((0, 2), float)
            V = np.zeros((0, 3))
            nvpm = V0.shape[0]

            for i in range(1):
                Vi = (v4 @ self.transforms[i].T)[:, : 3]
                cnt = i * nvpm
                V = np.vstack([V, Vi])
                J3 = np.linalg.det(self.transforms[i][:3, :3])
                J = np.abs(np.power(J3, 1 / 3))
                R = np.concatenate([R, np.copy(R0) * J])
                E = np.vstack((E, E0 + cnt))

            self.E_medial = E
            self.V_medial_rest = np.copy(V)
            self.V_medial = np.zeros_like(V)
            self.R_rest = np.copy(R)
            self.R = np.zeros_like(self.R_rest)

            self.V_medial[:] = self.V_medial_rest
            self.R[:] = self.R_rest

            self.n_medial = self.V_medial.shape[0]
