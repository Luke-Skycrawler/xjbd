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
        self.ps_target = None
        self.plane = None
        self.ray = None
        self.is_dragging = False

    def mouse_interact(self):
        # do mouse interactions
        io = gui.GetIO()
        cam_params = ps.get_view_camera_parameters()
        cam_pos = cam_params.get_position()
        cam_dir = cam_params.get_look_dir()
        

        # if io.MouseClicked[0]:
        self.is_dragging = gui.IsMouseDragging(0)
        if self.is_dragging and self.selected_handle != -1:
            screen_coords = io.MousePos
            world_ray = ps.screen_coords_to_world_ray(screen_coords)
            handle_pos = self.handle_pos[self.selected_handle]
            self.plane = (handle_pos, cam_dir)
            self.ray = (cam_pos, world_ray)
            ray_plane_intersection = self.compute_ray_plane_intersection(*self.ray, *self.plane)
            self.rod.pick_info.id = self.selected_handle
            self.rod.pick_info.target_pos = ray_plane_intersection
        else:
            self.rod.pick_info.id = -1
        if gui.IsMouseClicked(0):
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
            if self.selected_handle != -1: 
                ps.set_do_default_mouse_interaction(False)
            else: 
                ps.set_do_default_mouse_interaction(True)
            
            # set mouse plane 
            handle_pos = self.handle_pos[self.selected_handle]
            self.plane = (handle_pos, cam_dir)
            self.ray = (cam_pos, world_ray)
                
        # else: 
        #     self.selected_handle = -1  
        # self.visualize_updated_handles()
    # def update_handle_pos(self):
        
    #     self.handle_pos

    def compute_ray_plane_intersection(self, ray_origin, ray_dir, plane_point, plane_normal):
        dist = np.dot(plane_normal, plane_point - ray_origin)
        costheta = np.dot(ray_dir, plane_normal)
        return ray_origin + ray_dir * dist / costheta

    def draw_point(self, point, radius = None, color = (1, 0, 0)):
        ps_point_cloud = ps.register_point_cloud("target", np.array([point]), enabled= True, color = color)
        if radius is not None: 
            ps_point_cloud.add_scalar_quantity("radius", np.array([selected_handle_radius]))
            ps_point_cloud.set_point_radius_quantity("radius", autoscale=False)
        return ps_point_cloud

    def visualize_updated_handles(self):
        if self.selected_handle != -1:
            selected_handle_pos = self.handle_pos[self.selected_handle]
            selected_handle_radius = self.handle_radius[self.selected_handle] + 1e-3
            if self.ps_handle is None:
                self.ps_handle = ps.register_point_cloud("selected handle", np.array([selected_handle_pos]), enabled= True, radius = selected_handle_radius, color = (1, 0, 0))
                self.ps_handle.add_scalar_quantity("radius", np.array([selected_handle_radius]))
                self.ps_handle.set_point_radius_quantity("radius", autoscale=False)
            else:
                self.ps_handle.set_enabled(True)
                self.ps_handle.update_point_positions(np.array([selected_handle_pos]))
        elif self.ps_handle is not None:
            self.ps_handle.set_enabled(False)
            
        # if gui.IsMouseDragging(0):
        if self.is_dragging:
            
            # print(f"Dragging handle {self.selected_handle} at {self.ray[0]}")
            if self.plane is None or self.ray is None: 
                return
            ray_plane_intersection = self.compute_ray_plane_intersection(*self.ray, *self.plane)
            # print(f"Ray-plane intersection at {ray_plane_intersection}")
            if self.ps_target is None:
                # self.ps_target = ps.register_point_cloud("target", np.array([ray_plane_intersection]), enabled= True, color = (0, 1, 0))
                self.ps_target = self.draw_point(ray_plane_intersection, color = (0, 1, 0))
            else:
                self.ps_target.update_point_positions(np.array([ray_plane_intersection]))
                self.ps_target.set_enabled(True)
        elif self.ps_target is not None:
            self.ps_target.set_enabled(False)



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