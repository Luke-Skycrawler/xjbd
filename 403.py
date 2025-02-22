import numpy as np  
import warp as wp 
import igl
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import polyscope as ps
import polyscope.imgui as gui

dim = 3
skinning = "dqs"
assert skinning in ["dqs", "lbs", "lbs_matrix"]
@wp.kernel
def rot2quat(R: wp.array(dtype = wp.mat33), q: wp.array(dtype = wp.quat)):
    i = wp.tid()
    q[i] = wp.quat_from_matrix(R[i])
    q[i] /= wp.length(q[i])

class PSViewer: 
    def __init__(self, V, F, W, handles, BE):
        self.V = V
        self.F = F
        self.W = W
        self.C = handles
        self.BE = BE
        self.R = np.zeros((W.shape[1], 3, 3), dtype = float)

        self.q = wp.zeros((W.shape[1],), dtype = wp.quat)
        self.rotation = wp.zeros((W.shape[1]), dtype = wp.mat33)
        print("BE shape = ", BE.shape)
        self.ps_mesh = ps.register_surface_mesh("mesh", self.V, self.F)
        self.ps_skeleton = ps.register_curve_network("skeleton", self.C, BE, enabled = True, transparency= 1.0, material = "flat")
        ps.set_transparency_mode('simple')
        self.ui_handle = 0
        self.ui_rest_shape = False
        self.M = igl.lbs_matrix(self.V, self.W)
        self.T = np.zeros((self.M.shape[1], 3), dtype = float)

        ps.set_user_callback(self.callback)

        self.handles = []
        self.handle_pos = np.copy(handles).astype(float)
        self.selected_handles = []
        z3 = np.zeros((1, 3), dtype = float)
        i4 = np.identity(4, dtype = float)
        
        for i in range(handles.shape[0]):
            self.handles.append(ps.register_point_cloud(f"handle_{i}", z3))
            trans = i4
            trans[:3, 3] = handles[i]
            self.handles[i].set_transform(trans)
            # self.handles[i].set_transform_gizmo_enabled(True)

    def rotation_to_quats(self, R, q):
        wp.launch(rot2quat, dim = (self.W.shape[1]), inputs = [R, q])
    def callback(self):
        changed, self.ui_handle = gui.SliderInt("handle", self.ui_handle, 0, self.W.shape[1] - 1)
        changed, self.ui_rest_shape = gui.Checkbox("rest shape", self.ui_rest_shape)

        self.ps_mesh.add_scalar_quantity("BBW", self.W[:, self.ui_handle])

        for i, hi in enumerate(self.handles):
            if i >= self.W.shape[1]:
                break
            trans = hi.get_transform()
            self.handle_pos[i] = trans[:3, 3].astype(float)
            self.R[i] = trans[:3, :3]
            b = trans[:3, 3]
            b -= self.R[i] @ self.C[i]
            trans[: 3, 3] = b
            self.T[i * (dim + 1): (i + 1) * (dim + 1), :] = trans[:3, :].T
        

        x = np.zeros_like(self.V)

        if skinning == "dqs":
            # dual quaternion
            self.rotation.assign(self.R)
            self.rotation_to_quats(self.rotation, self.q)
            q = self.q.numpy().astype(np.float64)
            # dx = self.handle_pos - self.R @ self.C
            dx = np.zeros_like(self.handle_pos)
            for i in range(len(self.handles)):
                dx[i] = self.handle_pos[i] - self.R[i] @ self.C[i]
            # dx = self.handle_pos
            dx = dx.astype(np.float64)
            # # r = self.W @ q
            
            x = igl.dqs(self.V, self.W, q, dx)
        elif skinning == "lbs":
            # lbs
            for j in range(len(self.handles)):
                # lbs 
                xj = self.R[j] @ (self.V - self.C[j].reshape(-1, 3)).T
                xj += self.handle_pos[j].reshape(3, -1)
                xj *= self.W[:, j].reshape(1,- 1)
                x += xj.T
        elif skinning == "lbs_matrix":
            x = self.M @ self.T

        if self.ui_rest_shape:
            self.ps_mesh.update_vertex_positions(self.V)
        else:
            self.ps_mesh.update_vertex_positions(x)
        self.ps_skeleton.update_node_positions(self.handle_pos)
            
        io = gui.GetIO()
        cam_params = ps.get_view_camera_parameters()
        cam_pos = cam_params.get_position()
        
        if io.MouseClicked[0]:
            screen_coords = io.MousePos
            world_ray = ps.screen_coords_to_world_ray(screen_coords)
            dirs = self.handle_pos - cam_pos.reshape(-1, 3)
            norms = np.linalg.norm(dirs, axis = 1)
            dirs /= norms.reshape(-1, 1)
            
            diff = dirs - world_ray.reshape(1, 3)
            diff = np.linalg.norm(diff, axis = 1)
            handle_id = np.argmin(diff)
            bb_size = np.max(ps.get_bounding_box())
            if diff[handle_id] < 5e-2:
                for h in self.selected_handles:
                    self.handles[h].set_transform_gizmo_enabled(False)

                self.selected_handles = [handle_id]

            for i in self.selected_handles:
                self.handles[i].set_transform_gizmo_enabled(True)

            # world_pos = ps.screen_coords_to_world_position(screen_coords)

            # print(f"Click coords: {screen_coords}")
            # print(f"  world ray: {world_ray}")
            # print(f"  world pos: {world_pos}")
            # dir_c = cam_pos - world_pos
            # norm = np.linalg.norm(dir_c)
            # dir_c /= norm
            # print(f"dir calculated = {dir_c}")

def test_bbw(): 
    V, T, F = igl.read_mesh("assets/hand.mesh")
    C, BE, _, _, _, _ = igl.read_tgf("assets/hand.tgf")
    
    Q = igl.read_dmat("assets/hand-pose.dmat")

    # print(v.shape, e.shape, p.shape, PE.shape)
    print(BE.shape, C.shape, Q.shape, BE)
    # _, b, bc = igl.boundary_conditions(V,T, C, np.zeros(0, dtype = int), BE, np.zeros((0, 0), dtype = int), np.zeros((0, 0), dtype = int))
    _, b, bc = igl.boundary_conditions(V,T, C, np.arange(C.shape[0]), np.zeros((0, 2), dtype = int), np.zeros((0, 0), dtype = int), np.zeros((0, 0), dtype = int))

    bbw = igl.BBW(2, 8)
    W = bbw.solve(V, T, b, bc)
    print(W.shape)
    sum_W = np.sum(W, axis = 1)
    W /= sum_W.reshape(-1, 1)
    print(np.sum(W, axis = 1))
    print("BE shape = ", BE.shape)
    return V, F, W, C, BE

if __name__ == "__main__":
    ps.init()
    wp.init()
    V, F, W, C, BE = test_bbw()
    viewer = PSViewer(V, F, W, C, BE)
    ps.show()
    # print(p[0])