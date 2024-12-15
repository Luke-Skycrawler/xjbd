import warp as wp 
from utils.tobj import import_tobj
import igl

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
    i = I // ((n_yz + 1) ** 2)
    i_yz = I % ((n_yz + 1) ** 2)
    j = i_yz // (n_yz + 1)
    k = i_yz % (n_yz + 1)

    x = wp.vec3(float(i), float(j), float(k))
    return dx * x

@wp.func
def trans(I: int) -> int:
    i, j, k = I // 4, (I % 4) // 2, I % 2
    return (n_yz + 1) ** 2 * i + (n_yz + 1) * j + k 

@wp.func
def _trans(I: int) -> int:
    i = I // ((n_yz) ** 2)
    i_yz = I % ((n_yz) ** 2)
    j = i_yz // (n_yz)
    k = i_yz % (n_yz)
    return (n_yz + 1) ** 2 * i + (n_yz + 1) * j + k 


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
    centers[_e] = xc(e) + 0.5 * dx
    for i in range(2):
        for j in range(2):
            for k in range(2):
                I = e + i * (n_yz + 1) ** 2 + j * (n_yz + 1) + k
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
        wp.launch(init_elements, (n_elements, ), inputs = [self.centers, self.faces, self.indices, self.T, self.tet])
        wp.launch(init_nodes, self.xcs.shape, inputs = [self.xcs])


class TOBJLoader:
    def __init__(self):
        '''
        Before calling super().__init__(), make sure to define self.filename
        '''
        V, T = import_tobj(self.filename)
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
