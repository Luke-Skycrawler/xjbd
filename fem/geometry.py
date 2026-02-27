import warp as wp 
from utils.tobj import import_tobj
import igl
from typing import List
import numpy as np
L, W = 1, 0.2
mu, rho, lam = 1e6, 1., 125
g = 10.
n_x = 20
n_yz = 4
dx = L / n_x
nq = 8
wq = 1 / nq
wq_2d = 4
n_elements, n_nodes = n_yz ** 2 * n_x, (n_yz + 1) ** 2 * (n_x + 1)
n_tets = n_elements * 6
n_boundary_elements = n_yz * n_x * 4
delta = 0.08
volume = dx ** 3
area = dx ** 3
n_unknowns = n_nodes * 3


@wp.func
def xc(I: int) -> wp.vec3:
    '''
    coord of nodes
    '''
    i = I // ((n_yz + 1) * (n_yz + 1))
    i_yz = I % ((n_yz + 1) * (n_yz + 1))
    j = i_yz // (n_yz + 1)
    k = i_yz % (n_yz + 1)

    x = wp.vec3(float(i), float(j), float(k))
    return dx * x

@wp.func
def trans(I: int) -> int:
    i, j, k = I // 4, (I % 4) // 2, I % 2
    return (n_yz + 1) * (n_yz + 1) * i + (n_yz + 1) * j + k 

@wp.func
def _trans(I: int) -> int:
    i = I // ((n_yz) * (n_yz + 1))
    i_yz = I % ((n_yz) * (n_yz + 1))
    j = i_yz // (n_yz)
    k = i_yz % (n_yz)
    return (n_yz + 1) * (n_yz + 1) * i + (n_yz + 1) * j + k 


@wp.kernel
def init_faces_and_tets(faces: wp.array(dtype = wp.vec4i), tet: wp.array(dtype = wp.vec4i)):
    faces[0] = wp.vec4i(0, 1, 3, 2)
    faces[1] = wp.vec4i(4, 5, 1, 0)
    faces[2] = wp.vec4i(2, 3, 7, 6)
    faces[3] = wp.vec4i(4, 0, 2, 6)
    faces[4] = wp.vec4i(1, 5, 7, 3)
    faces[5] = wp.vec4i(5, 4, 6, 7)

    tet[0] = wp.vec4i(4, 1, 5, 3)
    tet[1] = wp.vec4i(5, 4, 3, 6)
    tet[2] = wp.vec4i(3, 5, 6, 7)
    tet[3] = wp.vec4i(0, 1, 4, 2)
    tet[4] = wp.vec4i(1, 4, 2, 3)
    tet[5] = wp.vec4i(4, 2, 3, 6)


@wp.kernel
def init_elements(centers: wp.array(dtype = wp.vec3), faces: wp.array(dtype = wp.vec4i), indices: wp.array(dtype = int), T: wp.array2d(dtype = int), tet: wp.array(dtype = wp.vec4i), nodes: wp.array2d(dtype = int)):
    _e = wp.tid()
    e = _trans(_e)
    centers[_e] = xc(e) + 0.5 * dx * wp.vec3(1.0)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                I = e + i * (n_yz + 1) * (n_yz + 1) + j * (n_yz + 1) + k
                J = i * 4 + j * 2 + k
                nodes[_e, J] = I

    for i in range(6):
        indices[_e * 36 + i * 6 + 0] = trans(faces[i][0]) + e
        indices[_e * 36 + i * 6 + 1] = trans(faces[i][1]) + e
        indices[_e * 36 + i * 6 + 2] = trans(faces[i][2]) + e
        indices[_e * 36 + i * 6 + 3] = trans(faces[i][2]) + e
        indices[_e * 36 + i * 6 + 4] = trans(faces[i][3]) + e
        indices[_e * 36 + i * 6 + 5] = trans(faces[i][0]) + e

    for i in range(6):
        for k in range(4):
            T[_e * 6 + i, k] = trans(tet[i][k]) + e


@wp.kernel
def init_nodes(xcs: wp.array(dtype = wp.vec3)):
    i = wp.tid()
    xcs[i] = xc(i)

'''
    tetrahedron geometry interface: 
        int: n_tets, n_nodes 
        wp.arrays: T, xcs, indices

    if the geometry is obj:
        int: n_faces, n_nodes
        wp.array: indices, xcs, 
        optional wp.array: u, v 
'''
class RodGeometryGenerator:
    def __init__(self):
        self.indices = wp.zeros((n_elements * 12 * 3), dtype = int)
        self.nodes = wp.zeros((n_elements, 8), dtype = int)
        self.centers = wp.zeros((n_elements), dtype = wp.vec3)
        self.xcs = wp.zeros((n_nodes), dtype = wp.vec3)
        self.faces = wp.zeros((6), dtype= wp.vec4i)
        self.boundary_centers = wp.zeros((n_boundary_elements), dtype = wp.vec3)
        self.T = wp.zeros((n_tets, 4), int)
        self.tet = wp.zeros((6), dtype = wp.vec4i)

        self.n_tets = self.T.shape[0]
        self.n_nodes = self.xcs.shape[0]
        self.geometry()
    
    def geometry(self):
        wp.launch(init_faces_and_tets, 0, inputs = [self.faces, self.tet])
        wp.launch(init_elements, (n_elements, ), inputs = [self.centers, self.faces, self.indices, self.T, self.tet, self.nodes])
        wp.launch(init_nodes, self.xcs.shape, inputs = [self.xcs])


class TOBJLoader:
    def __init__(self):
        '''
        Before calling super().__init__(), make sure to define self.filename
        '''
        if self.filename.endswith(".tobj"):
            V, T = import_tobj(self.filename)
        elif self.filename.endswith(".mesh"):
            V, T, _ = igl.read_mesh(self.filename)
            
        self.n_nodes = V.shape[0]
        self.n_tets = T.shape[0]
        self.xcs = wp.zeros((self.n_nodes), dtype = wp.vec3)
        self.T = wp.zeros((self.n_tets, 4), dtype = int)

        self.T.assign(T)
        self.xcs.assign(V)

        F = igl.boundary_facets(T)
        self.indices = wp.zeros(F.shape, dtype = int)
        self.indices.assign(F)
        print(f"{self.filename} loaded, {self.n_nodes} nodes, {self.n_tets} tets")


@wp.func
def plane_normal(v0: wp.vec3, v1: wp.vec3, v2: wp.vec3) -> wp.vec3:
    return wp.normalize(wp.cross(v1 - v0, v2 - v0))

@wp.kernel
def flip_face(verts: wp.array(dtype = wp.vec3), normals: wp.array(dtype = wp.vec3), indices: wp.array(dtype = int)):
    i = wp.tid()
    n = normals[i]
    i0 = indices[i * 3 + 0]
    i1 = indices[i * 3 + 1]
    i2 = indices[i * 3 + 2]
    ni = plane_normal(verts[i0], verts[i1], verts[i2])
    # if wp.dot(n, ni) < 0:
    if False:
        # flip v1 v2
        indices[i * 3 + 1] = i2
        indices[i * 3 + 2] = i1

@wp.kernel
def verify_normals(verts: wp.array(dtype = wp.vec3), normals: wp.array(dtype = wp.vec3), indices: wp.array(dtype = int)):
    i = wp.tid()
    i0 = indices[i * 3 + 0]
    i1 = indices[i * 3 + 1] 
    i2 = indices[i * 3 + 2]
    n = plane_normal(verts[i0], verts[i1], verts[i2])
    normals[i] = n

class TOBJComplex:
    def __init__(self):
        '''
        form a complex of all simulation meshes and exposes tet geometry interface 

        NOTE: need to have self.meshes_filename and self.transforms predefined before calling super().__init__()
        '''

        transforms = self.transforms 
        meshes_filename = self.meshes_filename

        self.n_nodes = 0
        self.n_tets = 0
        self.tet_start = []
        V = np.zeros((0, 3), dtype = float)
        T = np.zeros((0, 4), dtype = int)
        F_from_file = np.zeros((0, 3), dtype = int)
        F = np.zeros((0, 3), dtype = int)
        uv = np.zeros((0, 3), dtype = float)
        # only used as rest shape in cloth simulation

        while len(transforms) < len(meshes_filename):
            transforms.append(np.identity(4, dtype = float))
        
        assert(len(transforms) == len(meshes_filename))
        for f, trans in zip(meshes_filename, transforms):
            if f.endswith(".tobj"):
                v, t = import_tobj(f)
                ff = np.zeros((0, 3), int)
            elif f.endswith(".mesh"):
                v, t, _ = igl.read_mesh(f)
                ff = np.zeros((0, 3), int)
            elif f.endswith(".obj"):
                v, tc, _, ff, _, _ = igl.read_obj(f)
                t = np.zeros((0, 4), int)
                if tc is not None and tc.shape[0]:
                    if tc.shape[1] == 2:
                        tc = np.hstack((tc, np.zeros((tc.shape[0], 1), dtype = float)))
                    uv = np.vstack((uv, tc))
            
            v4 = np.ones((v.shape[0], 4), dtype = float)
            v4[:, :3] = v
            v = (v4 @ trans.T)[:, :3] 
            # V = np.vstack((V, v), dtype = float)
            V = np.vstack((V, v))
            if t.shape[0]:
                T = np.vstack((T, t + self.n_nodes))
            if ff.shape[0]:
                F_from_file = np.vstack([F_from_file, ff + self.n_nodes])

            self.tet_start.append(self.n_tets)
            self.n_nodes += v.shape[0]
            self.n_tets += t.shape[0]

        self.tet_start.append(self.n_tets)
        self.xcs = wp.zeros((self.n_nodes), dtype = wp.vec3) 
        self.xcs.assign(V)

        if uv.shape[0]:
            # u, v only defined for cloth  
            assert uv.shape[0] == self.n_nodes
            self.u = wp.zeros((self.n_nodes, ), dtype = float)
            self.v = wp.zeros((self.n_nodes, ), dtype = float)
            
            self.u.assign(uv[:, 0])
            self.v.assign(uv[:, 1])
            

        if T.shape[0]:
            self.T = wp.zeros((self.n_tets, 4), dtype = int)
            self.T.assign(T)
            FF = igl.boundary_facets(T)  
            FF, _ = igl.bfs_orient(FF)
            c, _ = igl.orientable_patches(FF)
            F, _ = igl.orient_outward(V, FF, c)
            
            assert(FF.shape[0] == F.shape[0])
        else: 
            # thin shell model 
            self.n_tets = F_from_file.shape[0]
            self.T = wp.zeros((self.n_tets, 3), dtype = int)
            print(f"F from file shape {F_from_file.shape}")
            self.T.assign(F_from_file)
            

        F = np.vstack([F, F_from_file])
        self.indices = wp.array(F.reshape(-1), dtype = int)
        self.n_faces = F.shape[0] 
        n0 = np.ones(3, dtype = float)
        self.N = N = igl.per_face_normals(V, F, n0)
        normals = wp.array(N, dtype = wp.vec3)
        
        # wp.launch(flip_face, (self.indices.shape[0] // 3,), inputs = [self.xcs, normals, self.indices])
        # wp.launch(verify_normals, (self.indices.shape[0] // 3,), inputs = [self.xcs, normals, self.indices])
        # print(f"after flipping: normals = {normals.numpy()}")
        print(f"{meshes_filename} loaded, {self.n_nodes} nodes, {self.n_tets} tets, {self.n_faces} triangles")