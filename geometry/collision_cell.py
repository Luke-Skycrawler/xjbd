import warp as wp 
import numpy as np
from fem.params import FEMMesh
from typing import List
import polyscope as ps
import igl
collision_eps = 0.01
PT_SET_SIZE = 4096
EE_SET_SIZE = 4096
FLT_MAX = 1e5
ZERO = 1e-6

@wp.struct
class TriangleSoup:
    '''
    a mesh containing all triangles in the simulation
    '''
    mesh_id: wp.uint64
    vertices: wp.array(dtype = wp.vec3)
    indices: wp.array(dtype = int)  # 1-d array of shape #F * 3, as per warp format 

@wp.struct
class CollisionList:
    a: wp.array(dtype = wp.vec2i)
    cnt: wp.array(dtype = int)

@wp.struct
class EdgeEdgeCollisionList:
    a: wp.array(dtype = wp.vec2i)
    cnt: wp.array(dtype = int)
    bary: wp.array(dtype = wp.vec3)

@wp.func
def append(cl: CollisionList, element: wp.vec2i):
    id = wp.atomic_add(cl.cnt, 0, 1)
    cl.a[id] = element

@wp.func
def append(cl: EdgeEdgeCollisionList, element: wp.vec2i, bary: wp.vec3):
    id = wp.atomic_add(cl.cnt, 0, 1)
    cl.a[id] = element
    cl.bary[id] = bary

@wp.kernel
def compute_inverted_vertex_single_mesh(x: wp.array(dtype = wp.vec3), geo: FEMMesh, Bm: wp.array(dtype = wp.mat33), inverted: wp.array(dtype = int)):
    e = wp.tid()
    t0 = x[geo.T[e, 0]]
    t1 = x[geo.T[e, 1]]
    t2 = x[geo.T[e, 2]]
    t3 = x[geo.T[e, 3]]
    
    Ds = wp.mat33(t0 - t3, t1 - t3, t2 - t3)
    
    F = Ds @ Bm[e]
    mesh_offset = 0
    if wp.determinant(F) < 0.0: 
        for i in range(4):
            inverted[geo.T[e, i] + mesh_offset] = 1        


@wp.kernel
def compute_inverted_vertex(x: wp.array(dtype = wp.vec3), tet_complex: wp.array2d(dtype = int), Bm: wp.array(dtype = wp.mat33), inverted: wp.array(dtype = int)):
    e = wp.tid()
    t0 = x[tet_complex[e, 0]]
    t1 = x[tet_complex[e, 1]]
    t2 = x[tet_complex[e, 2]]
    t3 = x[tet_complex[e, 3]]
    
    Ds = wp.mat33(t0 - t3, t1 - t3, t2 - t3)
    
    F = Ds @ Bm[e]
    if wp.determinant(F) < 0.0: 
        for i in range(4):
            inverted[tet_complex[e, i]] = 1        


def get_complex_size(meshes: List[FEMMesh], Fs: List[np.ndarray]):
    '''
    Fs: triangle list
    meshes: list of FEMMesh 
    '''
    tet_size = 0
    triangle_size = 0
    vert_size = 0
    for m, F in zip(meshes, Fs): 
        n_tets = m.T.shape[0]
        n_triangles = F.shape[0]
        n_points = m.xcs.shape[0]

        tet_size += n_tets
        triangle_size += n_triangles
        vert_size += n_points

    return tet_size, triangle_size, vert_size
    
@wp.func 
def point_triangle_distance_wp(v0: wp.vec3, v1: wp.vec3, v2: wp.vec3, v: wp.vec3):
    e1 = v1 - v0
    e2 = v2 - v0 
    e3 = v2 - v1
    
    n = wp.cross(e1, e2)
    na = wp.cross(v2 - v1, v - v1)
    nb = wp.cross(v0 - v2, v - v2)
    nc = wp.cross(v1 - v0, v - v0)
    
    barycentric = wp.vec3(
        wp.dot(n, na), wp.dot(n, nb), wp.dot(n, nc)
    ) / wp.length_sq(n)
    
    bary_abs = wp.abs(barycentric)
    bary_sum = bary_abs[0] + bary_abs[1] + bary_abs[2]

    ret = float(FLT_MAX)
    type = int(0)
    if bary_sum - 1.0 < ZERO:
        n_hat = wp.normalize(n)
        normal_distance = wp.dot(n_hat, v - v0)
        ret = wp.abs(normal_distance)
        type = 0
    else: 
        ev = v - v0
        ev3 = v - v1
        e1_hat = wp.normalize(e1)
        e2_hat = wp.normalize(e2)
        e3_hat = wp.normalize(e3)

        e1_dot = wp.dot(e1_hat, ev)
        e2_dot = wp.dot(e2_hat, ev) 
        e3_dot = wp.dot(e3_hat, ev3)
        
        e1_norm = wp.length(e1)
        e2_norm = wp.length(e2)
        e3_norm = wp.length(e3)

        edge_distances = wp.vec3(FLT_MAX)

        if e1_norm > e1_dot > 0.0:
            p1 = v0 + e1_hat * e1_dot
            edge_distances[0] = wp.length(v - p1)
        if e2_norm > e2_dot > 0.0:
            p2 = v0 + e2_hat * e2_dot
            edge_distances[1] = wp.length(v - p2)
        if e3_norm > e3_dot > 0.0:
            p3 = v1 + e3_hat * e3_dot
            edge_distances[2] = wp.length(v - p3)
            
        
        vertex_distances = wp.vec3(
            wp.length(v - v0),
            wp.length(v - v1),
            wp.length(v - v2)
        )

        v_min = wp.min(vertex_distances)
        # v_min = FLT_MAX
        e_min = wp.min(edge_distances)
        # e_min = FLT_MAX
        if v_min < e_min: 
            type = 1
        else: 
            type = int(wp.argmin(edge_distances)) + 2
        ret = min(v_min, e_min)

    return ret, type

@wp.func
def point_projects_inside_triangle(v0: wp.vec3, v1: wp.vec3, v2: wp.vec3, v: wp.vec3) -> bool:
    e1 = v1 - v0
    e2 = v2 - v0 
    e3 = v2 - v1
    
    n = wp.cross(e1, e2)
    na = wp.cross(v2 - v1, v - v1)
    nb = wp.cross(v0 - v2, v - v2)
    nc = wp.cross(v1 - v0, v - v0)
    
    barycentric = wp.vec3(
        wp.dot(n, na), wp.dot(n, nb), wp.dot(n, nc)
    ) / wp.length_sq(n)
    
    bary_abs = wp.abs(barycentric)
    bary_sum = bary_abs[0] + bary_abs[1] + bary_abs[2]
    
    return bary_sum - 1.0 < ZERO

@wp.func
def plane_normal(v0: wp.vec3, v1: wp.vec3, v2: wp.vec3) -> wp.vec3:
    return wp.normalize(wp.cross(v1 - v0, v2 - v0))

@wp.func
def inside_collision_cell(triangle_soup: TriangleSoup, neighbors: wp.array(dtype = int), tid: int, p: wp.vec3) -> bool: 
    ret = bool(True)
    v0 = triangle_soup.vertices[triangle_soup.indices[tid * 3]]
    v1 = triangle_soup.vertices[triangle_soup.indices[tid * 3 + 1]]
    v2 = triangle_soup.vertices[triangle_soup.indices[tid * 3 + 2]]

    v = wp.mat33(0.0)
    v[0] = v0
    v[1] = v1
    v[2] = v2
    
    n = plane_normal(v0, v1, v2)
    nn = wp.mat33(0.0)
    for x in range(3):
        tn = neighbors[tid * 3 + x]
        if tn != -1:
            vn0 = triangle_soup.vertices[tn * 3]
            vn1 = triangle_soup.vertices[tn * 3 + 1]
            vn2 = triangle_soup.vertices[tn * 3 + 2]

            nn[x] = plane_normal(vn0, vn1, vn2)
        else:
            ret = False
    
    if ret:
        for x in range(3):
            ne = wp.normalize(nn[x] + n) 
            
            eij = v[(x + 1) % 3] - v[x]
            neb = wp.cross(ne, eij)
            neb_hat = wp.normalize(neb)
            deplane = wp.dot(neb_hat, p - v[x])
            if deplane < 0.0:
                ret = False
    return ret

    

@wp.kernel
def point_triangle_collision(inverted: wp.array(dtype = int), triangles_soup: TriangleSoup, neighbors: wp.array(dtype = int), collision_list: CollisionList):
    i = wp.tid()
    
    if inverted[i]:
        pass
    else: 
        xi = triangles_soup.vertices[i]
        low = xi - wp.vec3(collision_eps)
        high = xi + wp.vec3(collision_eps)

        query = wp.mesh_query_aabb(triangles_soup.mesh_id, low, high)
        # iterates all triangles intersecting the dialated point volume
        y = int(0)
        # n_nodes = triangles_soup.vertices.shape[0]
        # for y in range(n_nodes):
        while wp.mesh_query_aabb_next(query, y):
            t0 = triangles_soup.indices[y * 3]
            t1 = triangles_soup.indices[y * 3 + 1]
            t2 = triangles_soup.indices[y * 3 + 2]
            
            if (inverted[t0] and inverted[t1] and inverted[t2]) or (t0 == i or t1 == i or t2 == i):
                # filter out inverted triangles and the 1-ring neighbor 
                pass
            else:
                xt0 = triangles_soup.vertices[t0]
                xt1 = triangles_soup.vertices[t1]
                xt2 = triangles_soup.vertices[t2]

                distance, _ = point_triangle_distance_wp(xt0, xt1, xt2, xi)
                element = wp.vec2i(i, y)
                if distance < collision_eps:
                    if point_projects_inside_triangle(xt0, xt1, xt2, xi) or inside_collision_cell(triangles_soup, neighbors, y, xi):
                        append(collision_list, element)


@wp.kernel
def edge_edge_collison(edges: wp.array(dtype = int), triangle_soup: TriangleSoup, edges_bvh: wp.uint64, collision_list: EdgeEdgeCollisionList):
    x = wp.tid()
    bary_closest = wp.vec3(0.0)
    closest = int(-1)
    closest_distance = float(FLT_MAX)

    ie0 = edges[x * 2 + 0]
    ie1 = edges[x * 2 + 1]
    
    e0 = triangle_soup.vertices[ie0]
    e1 = triangle_soup.vertices[ie1]

    low = wp.min(e0, e1) - wp.vec3(collision_eps)
    high = wp.max(e0, e1) + wp.vec3(collision_eps)

    query = wp.bvh_query_aabb(edges_bvh, low, high)
    y = int(0)

    while wp.bvh_query_next(query, y):
        # find closest other edge
        iy0 = edges[y * 2 + 0]
        iy1 = edges[y * 2 + 1]
        if iy0 == ie0 or iy0 == ie1 or iy1 == ie0 or iy1 == ie1:
            # skip if share a vertex
            pass
        else:
            v2 = triangle_soup.vertices[iy0]
            v3 = triangle_soup.vertices[iy1]

            bary = wp.closest_point_edge_edge(e0, e1, v2, v3, 1e-8)
            distance = bary[2]
            a = bary[0]
            b = bary[1]

            SKIP_EPS = 1e-4
            if distance > closest_distance or a < SKIP_EPS or a > 1.0 - SKIP_EPS or b < SKIP_EPS or b > 1.0 - SKIP_EPS:
                # skip if is close to an end vertex
                pass
            else: 
                closest_distance = distance
                closest = y
                bary_closest = bary

    inside_one_ring = False 
    if closest != -1 and not inside_one_ring and closest_distance < collision_eps:
        collision = wp.vec2i(x, closest)
        append(collision_list, collision, bary_closest)

@wp.kernel
def edge_aabbs(edges: wp.array(dtype = int), vertices: wp.array(dtype = wp.vec3), lowers: wp.array(dtype = wp.vec3), uppers: wp.array(dtype = wp.vec3)):
    i = wp.tid()
    e0 = vertices[edges[i * 2 + 0]]
    e1 = vertices[edges[i * 2 + 1]]
    lowers[i] = wp.min(e0, e1)  
    uppers[i] = wp.max(e0, e1)

def compute_point_triangle_collision():

    # __init__
    meshes = []
    Fs = []
    # placeholders 
    n_tets, n_triangles, n_nodes = get_complex_size(meshes, Fs)
    inverted = wp.zeros(shape = (n_nodes,) , dtype = int)
    Bm = wp.zeros(shape = (n_tets, ), dtype = wp.mat33)
    tet_complex = wp.zeros(shape = (n_tets, 4), dtype = int)
    triangle_soup = TriangleSoup()
    triangle_soup.indices = wp.zeros(shape = (n_triangles * 3,), dtype = int)
    triangle_soup.vertices = wp.zeros(shape = (n_nodes,), dtype = wp.vec3)
    triangle_soup.mesh_id = wp.Mesh(triangle_soup.vertices, triangle_soup.indices).id

    pt_set = CollisionList()
    pt_set.cnt = wp.zeros(shape = (1,), dtype = int)
    pt_set.a = wp.zeros(shape = (PT_SET_SIZE, ), dtype = int)

    wp.launch(compute_inverted_vertex, dim = (n_tets,), inputs = [triangle_soup.vertices, tet_complex, Bm, inverted])
    wp.launch(point_triangle_collision, dim = (n_nodes, ), inputs = [inverted, triangle_soup, pt_set])

class TestViewer:
    def __init__(self):
        
        n_triangles = 2
        n_nodes = 6

        rng = np.random.default_rng(123)
        rand_verts = rng.random(size = (n_nodes, 3), dtype = float)
        # fix the first triangle to x-z plane
        rand_verts[:3, 1] = 0.0
        triangle_soup = TriangleSoup()
        triangle_soup.indices = wp.array(np.arange(n_triangles * 3), dtype = int)
        triangle_soup.vertices = wp.zeros((n_nodes, ), dtype = wp.vec3)
        triangle_soup.vertices.assign(rand_verts)
        self.mesh_triangle_soup = wp.Mesh(triangle_soup.vertices, triangle_soup.indices)
        triangle_soup.mesh_id = self.mesh_triangle_soup.id
        self.triangle_soup = triangle_soup


        self.points = triangle_soup.vertices.numpy()
        self.V0 = np.copy(self.points[:n_triangles * 3, :])
        self.F = triangle_soup.indices.numpy().reshape((-1, 3))


        self.E = igl.edges(self.F).reshape(-1)
        self.n_edges = self.E.shape[0] // 2
        self.edges = wp.array(self.E, dtype = int)


        self.lowers = wp.zeros((self.n_edges, ), dtype = wp.vec3)
        self.uppers = wp.zeros_like(self.lowers)
        
        self.compute_edge_aabbs()
        self.edges_bvh = wp.Bvh(self.lowers, self.uppers)
        
        pt_set = CollisionList()
        pt_set.cnt = wp.zeros(shape = (1,), dtype = int)
        pt_set.a = wp.zeros(shape = (PT_SET_SIZE, ), dtype = wp.vec2i)


        ee_set = EdgeEdgeCollisionList()
        ee_set.cnt = wp.zeros(shape = (1,), dtype = int)        
        ee_set.a = wp.zeros(shape = (EE_SET_SIZE, ), dtype = wp.vec2i)
        ee_set.bary = wp.zeros(shape = (EE_SET_SIZE, 3), dtype = wp.vec3)


        inverted = wp.zeros(shape = (n_nodes,), dtype = int)

        self.inverted = inverted
        self.pt_set = pt_set
        self.ee_set = ee_set



        self.ps_mesh = ps.register_surface_mesh("triangle", self.V0[:3, ], self.F[:1])
        self.ps_mesh_fixed = ps.register_surface_mesh("triangle_fixed", self.V0[3:, ], self.F[1:] - 3)

        self.ps_mesh.set_transform_gizmo_enabled(True)
        # ps.set_user_callback(self.callback)
        ps.set_user_callback(self.callback_ee)

        self.n_nodes, self.n_triangles = n_nodes, n_triangles
        self.neighbors = wp.zeros((n_triangles * 3, ), dtype = int)
        self.neighbors.assign(np.array([-1] * n_triangles * 3, dtype = int))
        # FIXME: might have no neighbors if it is a cloth
        self.ps_points = ps.register_point_cloud("points", self.points)
        self.ps_points.set_radius(collision_eps)
        ps.show()

    def compute_edge_aabbs(self):
        wp.launch(edge_aabbs, dim = (self.n_edges,), inputs = [self.edges, self.triangle_soup.vertices, self.lowers, self.uppers])
        

    def callback_ee(self):
        self.move_geometry()

        # get updated uppers and lowers for edge aabbs  
        self.compute_edge_aabbs()
        self.edges_bvh.refit()

        self.ee_set.cnt.zero_()
        wp.launch(edge_edge_collison, dim = (self.n_edges,), inputs = [self.edges, self.triangle_soup, self.edges_bvh.id, self.ee_set])

        nee = self.ee_set.cnt.numpy()[0]
        if nee:
            self.ps_mesh.set_color((1.0, 0.0, 0.0))
        else: 
            self.ps_mesh.set_color((0.0, 0.0, 0.0))

    def move_geometry(self):
        trans = self.ps_mesh.get_transform()
        # only controls the first triangle here
        V0 = np.zeros((3, 4))
        V0[:, :3] = self.V0[: 3]
        V0[:, 3] = 1.0
        V = (V0 @ trans.T)[:, : 3]
        self.points[:3, :] = V

        # should all mean the same thing and upating one is equivalent to updating all 
        self.mesh_triangle_soup.points.assign(self.points)
        # self.triangle_soup.vertices.assign(self.points)

        self.ps_points.update_point_positions(self.points)
        
        

        self.mesh_triangle_soup.refit()
    
    def callback(self):
        self.move_geometry()
        self.pt_set.cnt.zero_()
        self.pt_set.a.zero_()
        wp.launch(point_triangle_collision, dim = (self.n_nodes, ), inputs = [self.inverted, self.triangle_soup, self.neighbors, self.pt_set])
        
        npt = self.pt_set.cnt.numpy()[0]
        # print(f"npt = {npt}")
        if npt:
            self.ps_mesh.set_color((1.0, 1.0, 0.0))
            # print("collision detected!")
        else:
            self.ps_mesh.set_color((0.0, 0.0, 0.0))

        
        
@wp.kernel
def test_distance_kernel(x: wp.array2d(dtype = wp.vec3), dist: wp.array(dtype = float), types: wp.array(dtype = int)):
    i = wp.tid()
    v0 = x[i, 0]
    v1 = x[i, 1]
    v2 = x[i, 2]
    v = x[i, 3]
    # dist[i], types[i] = point_triangle_distance_wp(v0, v1, v2, v)
    d, t = point_triangle_distance_wp(v0, v1, v2, v)
    dist[i] = d
    types[i] = t

        
        
        
def test():
    ps.init()
    viewer = TestViewer()

def test_distance():
    import ipctk
    from ipctk import edge_edge_distance, point_triangle_distance
    # import ipctk.point_triangle_distance as ptd
    n_samples = 1000

    x = wp.zeros((n_samples, 4), dtype = wp.vec3)

    rng = np.random.default_rng(123)
    rand_verts = rng.random(size = (n_samples, 4, 3,), dtype = float)
    x.assign(rand_verts)

    dist = wp.zeros((n_samples, ), dtype = float)
    types = wp.zeros((n_samples, ), dtype = int)    
    wp.launch(test_distance_kernel, (n_samples, ), inputs = [x, dist, types])
    dist_ref = np.zeros((n_samples, ), dtype = float)
    for i in range(n_samples):
        p = rand_verts[i, 3]
        v0 = rand_verts[i, 0]
        v1 = rand_verts[i, 1]
        v2 = rand_verts[i, 2]
        dist_ref[i] = point_triangle_distance(p, v0, v1, v2)
    
    dnp = dist.numpy() ** 2
    if n_samples <= 10:
        print(f"dist = {dnp}")
        print(f"ipctk ref = {dist_ref}")
    else:
        diff = dnp - dist_ref
        idx = np.arange(n_samples)[diff > 1e-5]
        big_diff = diff[idx]
        print(f"types = {types.numpy()[idx]}")
        print(f"big diff = {big_diff}, x = {rand_verts[idx, :, :]}")

if __name__ == "__main__":
    # test_distance()
    test()
    # compute_point_triangle_collision()

    