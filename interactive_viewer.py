import numpy as np 
from g2m.viewer import MedialViewer
import polyscope as ps
import polyscope.imgui as gui 

class MouseInteractionInterface: 
    '''
    super() should define the following attributes:
        handle_pos, handle_radius
    '''
    def __init__(self, *args):
        super().__init__(*args)
        self.selected_handle: int = -1
        
        self.ps_handle = None
    def mouse_interact(self):
        # do mouse interactions
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
            cand = diff < 1e-2
            cand_norms = norms[cand]
            if len(cand_norms):
                icand = np.argmin(cand_norms)
                n_handles = len(self.handle_pos)
                self.selected_handle = np.arange(n_handles)[cand][icand]
            else:
                self.selected_handle = -1

        # self.visualize_updated_handles()

    # def update_handle_pos(self):
        
    #     self.handle_pos

    def visualize_updated_handles(self):
        if self.selected_handle != -1:
            selected_handle_pos = self.handle_pos[self.selected_handle]
            selected_handle_radius = self.handle_radius[self.selected_handle] + 1e-3
            if self.ps_handle is None:
                self.ps_handle = ps.register_point_cloud("selected handle", np.array([selected_handle_pos]), enabled= True, radius = selected_handle_radius, color = (1, 0, 0))
                self.ps_handle.add_scalar_quantity("radius", np.array([selected_handle_radius]))
                self.ps_handle.set_point_radius_quantity("radius", autoscale=False)
            else:
                self.ps_handle.update_point_positions(np.array([selected_handle_pos]))
        else:
            self.ps_handle = None


class MouseInteractionSocket(MedialViewer):
    def __init__(self, rod, static_mesh = None):
        super().__init__(rod, static_mesh)
        self.handle_pos = self.V_medial
        self.handle_radius = self.R

class InteractiveMedialViewer(MouseInteractionInterface, MouseInteractionSocket):
    def __init__(self, rod, static_mesh = None):
        super().__init__(rod, static_mesh)
    
        
    def callback(self):
        self.mouse_interact()
        return super().callback()

    def update_medial(self):
        super().update_medial()
        self.visualize_updated_handles()