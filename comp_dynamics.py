import igl 
import numpy as np

import polyscope as ps
import polyscope.imgui as gui

def load():
    
    # V, _, _, F, _, _ = igl.read_obj("assets/elephant.obj")
    # default obj
    C, BE, _, _, _, _ = igl.read_tgf("assets/elephant_handles.tgf")

    # default weight and handles

    # C, BE, _, _, _, _ = igl.read_tgf("assets/elephant.tgf")
    # W = igl.read_dmat("assets/elephant-weights.dmat")
    
    T = igl.read_dmat("assets/elephant-anim.dmat")
    verts, tets, _= igl.read_mesh("assets/elephant.mesh")

    # deprecated code to find closest vertices to control points
    # updated handles are stored in `elephant_handles.tgf`

    # cid = []
    # for c in C:
    #     distances = np.linalg.norm(verts - c.reshape(1, 3), axis = 1)
    #     idx = np.argmin(distances)
    #     cid.append(idx)
    # cid = np.array(cid)
    # print(cid)
    # print(verts[cid])

    boundary = igl.boundary_facets(tets)
    V = verts
    F = boundary

    # bone-based weights
    ok, b, bc = igl.boundary_conditions(verts, tets, C, np.zeros(0, dtype = int), BE, np.zeros((0, 0), dtype = int), np.zeros((0, 0), dtype = int))
    print(ok)
    bbw = igl.BBW(2, 8)
    W = bbw.solve(V, tets, b, bc)
    W_sum = np.sum(W, axis = 1)
    W = W / W_sum.reshape(-1, 1)
    return V, F, W, C, BE, T

class PSViewer:
    def __init__(self, V, F, W, T, C, BE):
        self.V = V
        self.F = F
        self.W = W
        self.T = T
        self.C = C
        self.BE = BE

        self.M = igl.lbs_matrix(V, W)

        self.ps_mesh = ps.register_surface_mesh("mesh", self.V, self.F)
        ps.set_user_callback(self.callback)

        self.frame = 0
        self.n_frames = T.shape[1]
        self.skeleton = ps.register_curve_network("skeleton", C, BE)

        self.ui_rest = False
    def callback(self):
        changed, self.ui_rest = gui.Checkbox("Rest", self.ui_rest)
        Tf = self.T[:, self.frame].reshape(3, -1).T
        Vf = self.M @ Tf

        if self.ui_rest:
            self.ps_mesh.update_vertex_positions(self.V)
        else :
            self.ps_mesh.update_vertex_positions(Vf)
        
        self.frame += 1
        self.frame = self.frame % self.n_frames
        

if __name__ == "__main__":
    ps.init()
    np.printoptions(precision = 3)
    V, F, W, C, BE, T = load()
    viewer = PSViewer(V, F, W, T, C, BE)
    ps.show()
        