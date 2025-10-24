import numpy as np
from scipy.spatial.transform import Rotation as R

class SceneReader:
    def __init__(self, config_file: str = "scenes/11_erleben/cubeCliffCO.txt"):
        '''
        Read IPC style txt config file
        '''
        self.configs_file = config_file
        
        self.meshes_filenames = []
        self.transforms = [] 
        self.static_object = []
        self.static_transform = None
        self.read()

        self.transforms = np.array(self.transforms, dtype = float)
    
    def compose_transform(self, translate, rotation_euler, scale):
        '''
        scale, rotate, translate
        '''
        s = np.diag(np.append(scale, 1.0))
        r = R.from_euler('xyz', rotation_euler).as_matrix()
        r4 = np.eye(4, dtype = float)
        r4[:3, :3] = r
        t = r4 @ s
        t[:3, 3] = translate
        return t

    def read(self):
        with open(self.configs_file, 'r') as file:
            for line in file: 
                parts = line.split()
                if not parts:
                    continue

                if parts[0] == "shapes":
                    n_meshes = int(parts[2])
                    for _ in range(n_meshes):
                        mesh_line = next(file).strip()
                        mesh_parts = mesh_line.split()
                        mesh_filename = mesh_parts[0]
                        self.meshes_filenames.append(mesh_filename)
                        
                        # the next 9 numbers: 
                        # translation (3), rotation (euler angle in degree), scale (3)
                        translate = np.array([float(mesh_parts[1]), float(mesh_parts[2]), float(mesh_parts[3])], dtype = float)

                        rotation_euler = np.deg2rad(np.array([float(mesh_parts[4]), float(mesh_parts[5]), float(mesh_parts[6])], dtype = float))
                        scale = np.array([float(mesh_parts[7]), float(mesh_parts[8]), float(mesh_parts[9])], dtype = float)

                        self.transforms.append(self.compose_transform(translate, rotation_euler, scale))


                elif parts[0] == "meshCO":
                    self.static_object.append(parts[1])    
                    # 3 translation + 1 scale
                    translate = np.array([float(parts[2]), float(parts[3]), float(parts[4])], dtype = float)
                    self.static_transform = np.array([self.compose_transform(translate, np.array([0.0, 0.0, 0.0], dtype = float), np.array([float(parts[5]), float(parts[5]), float(parts[5])], dtype = float))])


