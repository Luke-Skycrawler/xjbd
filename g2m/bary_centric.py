from utils.tobj import import_tobj, export_tobj
from g2m.medial import SlabMesh, slabmesh_default
import numpy as np 
import igl
class TetBaryCentricCompute:
    def __init__(self, model, nv = None):
        
        self.V, self.T = import_tobj(f"assets/{model}.tobj")
        
        if nv is None:

            self.slabmesh = slabmesh_default()
        else :
            self.slabmesh = SlabMesh(f"data/{model}_v{nv}.ma")

        self.bc, self.tids = None, None
        if self.V is not None:
            self.embed(self.V)
        
    def reload(self, filename):
        self.slabmesh = SlabMesh(filename)
        # if self.V is not None:
        #     self.embed(self.V)



    def embed(self, V):
         
        bc = []
        tids = []
        self.a = V[self.T[:, 0]]
        self.b = V[self.T[:, 1]]
        self.c = V[self.T[:, 2]]
        self.d = V[self.T[:, 3]]
        for point in self.slabmesh.V: 
            bary, tid = self.barycentric_coord(point)
            if bary is None:
                continue
            bc.append(bary)
            tids.append(tid)
            # print(bary, tid)
        
        if len(bc): 
            self.bc = np.array(bc)
            self.tids = np.array(tids)


    def barycentric_coord(self, point):
        T, a, b, c, d = self.T, self.a, self.b, self.c, self.d

        p = np.zeros((T.shape[0], 3))
        p[:] = point
        bary = igl.barycentric_coordinates_tet(p, a, b, c, d)
        bary_max = np.max(bary, axis = 1)
        bary_min = np.min(bary, axis = 1)
        select= (1 >= bary_max) & (bary_min >= 0.0)
        
        if (select.any()):
            tid = np.argwhere(select)[0, 0]
            bary = bary[tid]
        else: 
            tid = None
            bary = None
        return bary, tid

    def deform(self, Qi):
        if self.bc is None:
            return
        V = self.V + Qi.reshape(-1, 3)
        T = self.T
        for i, tid in enumerate(self.tids):
            self.slabmesh.V[i] = V[T[tid]].T @ self.bc[i]

    def compute_weight(self, Wp):
        if self.bc is None:
            return 
        T = self.T
        W = np.zeros((self.slabmesh.V.shape[0], Wp.shape[1]), float)
        for i, tid in enumerate(self.tids):
            W[i] = Wp[T[tid]].T @ self.bc[i]
        return W

if __name__ == "__main__":
    tbtt = TetBaryCentricCompute("bug", 30)