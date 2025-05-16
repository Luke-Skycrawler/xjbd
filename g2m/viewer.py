from stretch import PSViewer
import polyscope as ps
import polyscope.imgui as gui
import numpy as np
class MedialViewerInterface:
    '''
    super() should define the following attributes:
        V_medial, R, V_rest, R_rest, E
    '''
    def __init__(self, *args): 
        super().__init__(*args)
        self.register_medial()
        
    def register_medial(self):

        self.ps_medial = ps.register_curve_network("medial", self.V_medial, self.E)
        self.ps_spheres = ps.register_point_cloud("spheres", self.V_medial)
        self.ps_spheres.add_scalar_quantity("radius", self.R)   
        self.ps_spheres.set_point_radius_quantity("radius", autoscale=False)


    def update_medial(self):
        self.ps_medial.update_node_positions(self.V_medial)
        self.ps_spheres.update_point_positions(self.V_medial)
        self.ps_spheres.add_scalar_quantity("radius", self.R)
        self.ps_spheres.set_point_radius_quantity("radius", autoscale=False)

class MedialViewerSocket(PSViewer):
    def __init__(self, rod, static_mesh):
        super().__init__(rod, static_mesh)
        n_modes, n_nodes = 12, 30
        

        self.V_medial = self.rod.V_medial
        self.R = self.rod.R

        self.V_rest = self.rod.V_medial_rest
        self.R_rest = self.rod.R_rest

        self.E = rod.E_medial

        np.save("output/medial/E.npy", self.E)
    
class MedialViewer(MedialViewerInterface, MedialViewerSocket):
    def __init__(self, rod, static_mesh = None):
        super().__init__(rod, static_mesh)

    def update_medial(self):
        # V, R = self.get_VR()
        V, R = self.rod.get_VR()

        # V, R is binded to the controlling arrays by reference
        self.V_medial[:] = V
        self.R[:] = R
        super().update_medial()

    def save(self):
        super().save()
        # np.save(f"output/medial/V{self.frame:04d}.npy", self.V_medial)
        # np.save(f"output/medial/R{self.frame:04d}.npy", self.R)

    def callback(self):
        super().callback()
        self.update_medial()
        