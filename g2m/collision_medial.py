import warp as wp
import numpy as np
from geometry.collision_cell import TriangleSoup, point_triangle_distance_wp, point_projects_inside_triangle, inside_collision_cell
from geometry.static_scene import StaticScene
from mtk import MedialSphere, ConeConeConstraint, SlabSphereConstraint
from scipy.sparse import bsr_array, bsr_matrix
from warp.sparse import bsr_set_from_triplets, bsr_zeros
from g2m.analyze import compute_distance_cone_cone, compute_distance_slab_sphere
from typing import Tuple, Set
import igl
CC_SET_SIZE = 1024
SS_SET_SIZE = 1024
G_SET_SIZE = 1024

ground_rel_stiffness = 10

per_mesh_verts = 211
COLLISION_DEBUG = False

cc_collision = True
ss_colliison = True
sg_collision = True
cc_static_collision = True

# @wp.kernel
# def collision_medial(edges: wp.array(dtype = wp.vec2i), vertices: wp.array(dtype = wp.vec3), radius: wp.array(dtype = float), ee_set: EdgeEdgeCollisionList):
#     i, j = wp.tid()
#     if i < j: 
#         # brute force edge-edge collision detection

@wp.struct
class MedialGeometry: 
    vertices: wp.array(dtype = wp.vec3)
    radius: wp.array(dtype = float)
    edges: wp.array(dtype = wp.vec2i)
    faces: wp.array(dtype = wp.vec3i)
    connectivity: wp.array(dtype = int)

@wp.struct
class ConeConeCollisionList:
    a: wp.array(dtype = wp.vec4i)
    dist: wp.array(dtype = float)
    E: wp.array(dtype = float)
    cnt: wp.array(dtype = int)

@wp.struct
class SlabSphereCollisionList:
    a: wp.array(dtype = wp.vec4i)
    cnt: wp.array(dtype = int)
    E: wp.array(dtype  = float)
    # dist: wp.array(dtype = float)

@wp.struct
class SphereGroundCollisionList:
    a: wp.array(dtype = int)
    cnt: wp.array(dtype = int)
    E: wp.array(dtype = float)
    hr: wp.array(dtype = wp.vec2)

@wp.struct
class StaticCollisionList:  
    a: wp.array(dtype = wp.vec2i)
    cnt: wp.array(dtype = int)
    E: wp.array(dtype = float)  

@wp.func
def append(static_list: StaticCollisionList, element: wp.vec2i, dist_sqr: float):    
    id = wp.atomic_add(static_list.cnt, 0, 1)
    static_list.a[id] = element
    wp.atomic_add(static_list.E, 0, dist_sqr * dist_sqr)    


@wp.func
def append(cc_list: ConeConeCollisionList, element: wp.vec4i, dist: float):
    id = wp.atomic_add(cc_list.cnt, 0, 1)
    cc_list.a[id] = element
    cc_list.dist[id] = dist
    wp.atomic_add(cc_list.E, 0, dist * dist)
    # wp.atomic_add(cc_list.E, 0, dist)

@wp.func
def append(ss_list: SlabSphereCollisionList, element: wp.vec4i, dist: float):
    id = wp.atomic_add(ss_list.cnt, 0, 1)
    ss_list.a[id] = element
    wp.atomic_add(ss_list.E, 0, dist * dist)
    # ss_list.dist[id] = dist

@wp.func
def append(g_list: SphereGroundCollisionList, element: int, hr: wp.vec2, dist: float):
    id = wp.atomic_add(g_list.cnt, 0, 1)
    g_list.a[id] = element
    g_list.hr[id] = hr

    wp.atomic_add(g_list.E, 0, dist * dist)

@wp.func
def is_1_ring(a: int, b: int, c: int, d:int):
    return a == c or a == d  or b == c or b == d

@wp.func
def to_hash(x: int, y: int, geo: MedialGeometry):
    i = wp.min(x, y)
    j = wp.max(x, y)
    n_verts = geo.vertices.shape[0]
    return i * n_verts + j

@wp.func
def is_2_ring(geo: MedialGeometry, ea0: int, ea1: int, eb0: int, eb1: int):
    return is_connected(geo, ea0, eb0) or is_connected(geo, ea0, eb1) or is_connected(geo, ea1, eb0) or is_connected(geo, ea1, eb1)    

@wp.func
def is_connected(geo: MedialGeometry, x: int, y: int):
    h = to_hash(x, y, geo)
    f = wp.lower_bound(geo.connectivity, 0, geo.connectivity.shape[0], h)
    return h == geo.connectivity[f]
    
def to_hash_np(x, y, n_vertices):
    i = np.minimum(x, y)
    j = np.maximum(x, y)
    return i * n_vertices + j

@wp.kernel
def cone_cone_collision_set(geo: MedialGeometry, cc_list: ConeConeCollisionList, body: wp.array(dtype = int)):
    i, j = wp.tid()
    if i < j:
        ex = geo.edges[i]
        ey = geo.edges[j]
        c0 = geo.vertices[ex[0]]
        c1 = geo.vertices[ex[1]]
        c2 = geo.vertices[ey[0]]
        c3 = geo.vertices[ey[1]]
        r0 = geo.radius[ex[0]]
        r1 = geo.radius[ex[1]]
        r2 = geo.radius[ey[0]]
        r3 = geo.radius[ey[1]]
        dist, _, foo, bar = compute_distance_cone_cone(c0, c1, c2, c3, r0, r1, r2, r3)
        # hack = ex[0] // per_mesh_verts == ey[0] // per_mesh_verts
        hack = body[ex[0]] == body[ey[0]]
        refuse_cond = is_1_ring(ex[0], ex[1], ey[0], ey[1]) or is_2_ring(geo, ex[0], ex[1], ey[0], ey[1]) or hack
        if dist < 0.0 and not refuse_cond:
            col = wp.vec4i(ex[0], ex[1], ey[0], ey[1])
            append(cc_list, col, dist)
@wp.kernel
def cone_cone_collision_set_static(geo: MedialGeometry, geo_static: MedialGeometry, cc_list: ConeConeCollisionList):
    i, j = wp.tid()
    ex = geo.edges[i]
    c0 = geo.vertices[ex[0]]
    c1 = geo.vertices[ex[1]]
    r0 = geo.radius[ex[0]]
    r1 = geo.radius[ex[1]]

    ey = geo_static.edges[j]
    c2 = geo_static.vertices[ey[0]]
    c3 = geo_static.vertices[ey[1]]
    r2 = geo_static.radius[ey[0]]
    r3 = geo_static.radius[ey[1]]
    dist, _, foo, bar = compute_distance_cone_cone(c0, c1, c2, c3, r0, r1, r2, r3)

    if dist < 0.0:
        col = wp.vec4i(ex[0], ex[1], ey[0], ey[1])
        append(cc_list, col, dist)

@wp.kernel
def slab_sphere_collision_set(geo: MedialGeometry, ss_list: SlabSphereCollisionList, body: wp.array(dtype = int)):
    i, j = wp.tid()
    slab = geo.faces[i]
    b = geo.vertices[j]

    c0 = geo.vertices[slab[0]]
    c1 = geo.vertices[slab[1]]
    c2 = geo.vertices[slab[2]]

    r0 = geo.radius[slab[0]]
    r1 = geo.radius[slab[1]]
    r2 = geo.radius[slab[2]]

    rb = geo.radius[j]

    dist, _, foo, bar = compute_distance_slab_sphere(c0, c1, c2, b, r0, r1, r2, rb)
    # hack = j // per_mesh_verts == slab[0] // per_mesh_verts
    hack = body[j] == body[slab[0]]
    refuse_cond = slab[0] == j or slab[1] == j or slab[2] == j or hack
    if dist < 0.0 and not refuse_cond:
        col = wp.vec4i(slab[0], slab[1], slab[2], j)
        # append(ss_list, col, dist * dist)
        append(ss_list, col, dist)

@wp.kernel
def sphere_ground_set(geo: MedialGeometry, g_list: SphereGroundCollisionList, ground_plane: float):
    i = wp.tid()
    b = geo.vertices[i]
    r = geo.radius[i]
    d = (b[1] - ground_plane)
    dist = d - r
    if dist < 0.0:
        hr = wp.vec2(d, r)
        append(g_list, i, hr, dist * dist)

@wp.kernel
def sphere_static_collision(geo: MedialGeometry, triangles_soup: TriangleSoup, collision_list: StaticCollisionList, neighbors: wp.array(dtype = int)):
    '''
    sphere collision with static triangle soup
    '''
    i = wp.tid()
    
    xi = geo.vertices[i]
    ri = geo.radius[i]
    low = xi - wp.vec3(ri)
    high = xi + wp.vec3(ri)

    query = wp.mesh_query_aabb(triangles_soup.mesh_id, low, high)
    # iterates all triangles intersecting the dialated point volume
    y = int(0)
    # n_triangles = triangles_soup.indices.shape[0] // 3
    # for y in range(n_triangles):
    while wp.mesh_query_aabb_next(query, y):
        t0 = triangles_soup.indices[y * 3]
        t1 = triangles_soup.indices[y * 3 + 1]
        t2 = triangles_soup.indices[y * 3 + 2]
        

        xt0 = triangles_soup.vertices[t0]
        xt1 = triangles_soup.vertices[t1]
        xt2 = triangles_soup.vertices[t2]

        distance, _ = point_triangle_distance_wp(xt0, xt1, xt2, xi)
        element = wp.vec2i(i, y)
        if distance < ri:
            n = plane_normal_wp(xt0, xt1, xt2)
            if wp.dot(n, xi - xt0) < 0.0:
                distance = -distance
            dmr = distance - ri
            if point_projects_inside_triangle(xt0, xt1, xt2, xi) or inside_collision_cell(triangles_soup, neighbors, y, xi):
                append(collision_list, element, dmr * dmr)

@wp.kernel
def refuse_2_ring(geo: MedialGeometry, cc_list: ConeConeCollisionList):
    i = wp.tid()
    if i < cc_list.cnt[0]:
        a = cc_list.a[i]
        e0 = a[0]
        e1 = a[1]
        e2 = a[2]
        e3 = a[3]
        if not is_2_ring(geo, e0, e1, e2, e3):
            dist = cc_list.dist[i]
            wp.atomic_add(cc_list.E, 0, dist * dist)
        
@wp.func
def plane_normal_wp(v0: wp.vec3, v1: wp.vec3, v2: wp.vec3) -> wp.vec3:
    return wp.normalize(wp.cross(v1 - v0, v2 - v0))

def plane_normal(v0, v1, v2):
    n = np.cross(v1 - v0, v2 - v0)
    return n / np.sqrt(np.dot(n, n))    

class MedialCollisionDetector:
    def __init__(self, V_medial, R, E, F, body_medial, ground = None, dense = True, static_objects: StaticScene = None): 
        '''
        medial-medial collision detection & response
        '''

        # numpy arrays
        self.V = V_medial
        self.V_rest = np.copy(V_medial)
        self.R = R
        self.E = E
        self.F = F
        self.B = body_medial
        self.body = wp.array(body_medial, dtype = int) 

        self.cc_set = []
        self.ss_set = []

        self.ground = ground
        # warp arrays 
        self.medial_geo = MedialGeometry()
        self.indices_set:set[int] = set()
        
        self.medial_geo.vertices = wp.array(V_medial, dtype = wp.vec3)
        self.medial_geo.radius = wp.array(R, dtype = float)
        self.medial_geo.edges = wp.array(E, dtype = wp.vec2i)
        self.medial_geo.faces = wp.array(F, dtype = wp.vec3i)
        n_verts= V_medial.shape[0]
        conn = to_hash_np(E[:, 0], E[:, 1], n_verts)
        conn = np.sort(conn)
        self.medial_geo.connectivity = wp.array(conn, dtype = int)
        

        self.n_vertices = V_medial.shape[0]
        self.n_edges = E.shape[0]
        self.n_faces = F.shape[0]

        self.ee_set = ConeConeCollisionList()
        self.ee_set.a = wp.zeros(CC_SET_SIZE, dtype = wp.vec4i)
        self.ee_set.cnt = wp.zeros(1, dtype = int)
        self.ee_set.E = wp.zeros(1, dtype = float)
        self.ee_set.dist = wp.zeros(CC_SET_SIZE, dtype = float)

        self.pt_set = SlabSphereCollisionList()
        self.pt_set.a = wp.zeros(SS_SET_SIZE, dtype = wp.vec4i)
        self.pt_set.cnt = wp.zeros(1, dtype = int)
        self.pt_set.E = wp.zeros(1, dtype = float)
        # self.pt_set.dist = wp.zeros(CC_SET_SIZE, dtype = float)

        
        self.g_set = SphereGroundCollisionList()
        self.g_set.a = wp.zeros(G_SET_SIZE, dtype = int)
        self.g_set.cnt = wp.zeros(1, dtype = int)
        self.g_set.E = wp.zeros(1, dtype = float)
        self.g_set.hr = wp.zeros(G_SET_SIZE, dtype = wp.vec2)
        
        vv = lambda e: (max(e[0], e[1]), min(e[0], e[1])) 
        self.vv_adjacency = set([vv(e) for e in E])
        self.dense = dense
        
        self.rest_cc: Set[Tuple[int, int, int, int]] = set()
        self.rest_pt: Set[Tuple[int, int, int, int]] = set()
        # self.get_rest_collision_set()
        if static_objects is not None:
            self.define_static_mesh(static_objects)

    def define_static_mesh(self, static_objects: StaticScene):
        self.static_objects = static_objects
        self.staic_triangles = wp.Mesh(static_objects.xcs, static_objects.indices, )
        static_soup = TriangleSoup()
        static_soup.indices = static_objects.indices
        static_soup.vertices = static_objects.xcs
        static_soup.mesh_id = self.staic_triangles.id
        self.static_soup = static_soup
        
        p_static_set = StaticCollisionList()
        p_static_set.cnt = wp.zeros((1, ), dtype =int)
        p_static_set.a = wp.zeros((CC_SET_SIZE, ), dtype = wp.vec2i)
        p_static_set.E = wp.zeros((1, ), dtype = float)

        self.p_static_set = p_static_set
        self.n_static = 0


        self.static_indices = static_objects.indices.numpy()
        self.static_V = static_objects.xcs.numpy()
        
        TT, _ = igl.triangle_triangle_adjacency(self.static_indices.reshape(-1, 3))
        self.static_neighbors = wp.array(TT.reshape(-1), dtype = int)

        if static_objects.has_medials:
            self.ee_static_set = ConeConeCollisionList()
            self.ee_static_set.a = wp.zeros(CC_SET_SIZE, dtype = wp.vec4i)
            self.ee_static_set.cnt = wp.zeros(1, dtype = int)
            self.ee_static_set.E = wp.zeros(1, dtype = float)
            self.ee_static_set.dist = wp.zeros(CC_SET_SIZE, dtype = float)
            # define medial geometry for static objects
            self.medial_geo_static = MedialGeometry()
            self.medial_geo_static.vertices = wp.array(static_objects.V_medial, dtype = wp.vec3)
            self.medial_geo_static.radius = wp.array(static_objects.R, dtype = float)
            self.medial_geo_static.edges = wp.array(static_objects.E_medial, dtype = wp.vec2i)
            self.medial_geo_static.faces = wp.array(static_objects.F_medial, dtype = wp.vec3i)
            n_verts= static_objects.V_medial.shape[0]
            conn = to_hash_np(static_objects.E_medial[:, 0], static_objects.E_medial[:, 1], n_verts)
            conn = np.sort(conn)
            self.medial_geo_static.connectivity = wp.array(conn, dtype = int)
            self.n_edges_static = static_objects.E_medial.shape[0]

    def get_rest_collision_set(self):
        self.collision_set(self.V)
        rest_pt = [(i[0], i[1], i[2], i[3]) for i in self.ss_id]
        rest_cc = [(i[0], i[1], i[2], i[3]) for i in self.cc_id]
        self.rest_pt.update(rest_pt)
        self.rest_cc.update(rest_cc)

    def refit(self, V, R = None):
        if R is not None:
            self.R[:] = R
            self.medial_geo.radius.assign(R)
        self.V[:] = V
        self.medial_geo.vertices.assign(V)

    def is_1_ring(self, e0, e1, e2, e3):
        return e0 == e2 or e0 == e3 or e1 == e2 or e1 == e3

    
    def is_connected(self, v0, v1):
        key = (max(v0, v1), min(v0, v1))
        return key in self.vv_adjacency
            
    def is_2_ring(self, e0, e1, e2, e3):
        return self.is_connected(e0, e2) or self.is_connected(e0, e3) or self.is_connected(e1, e2) or self.is_connected(e1, e3)

    def collision_set(self, V, R = None, energy_only = False):
        self.refit(V, R)
        ret = 0.0
        if ss_colliison:
            self.pt_set.cnt.zero_()
            self.pt_set.E.zero_()
            wp.launch(slab_sphere_collision_set, (self.n_faces, self.n_vertices), inputs= [self.medial_geo, self.pt_set, self.body])
            ret += self.pt_set.E.numpy()[0]

        if cc_collision:
            self.ee_set.cnt.zero_()
            self.ee_set.E.zero_()
            wp.launch(cone_cone_collision_set, (self.n_edges, self.n_edges), inputs = [self.medial_geo, self.ee_set, self.body])
            ret += self.ee_set.E.numpy()[0]


        if self.ground is not None and sg_collision:
            self.g_set.cnt.zero_()
            self.g_set.E.zero_()
            wp.launch(sphere_ground_set, self.n_vertices, inputs = [self.medial_geo, self.g_set, self.ground])
            ret += self.g_set.E.numpy()[0] * ground_rel_stiffness

        if hasattr(self, "static_soup"):
            if sg_collision:
                self.p_static_set.cnt.zero_()
                self.p_static_set.E.zero_()
                wp.launch(sphere_static_collision, self.n_vertices, inputs = [self.medial_geo, self.static_soup, self.p_static_set, self.static_neighbors])
                ret += self.p_static_set.E.numpy()[0] * ground_rel_stiffness

            if self.static_objects.has_medials and cc_static_collision:
                # only cone-cone collision
                self.ee_static_set.cnt.zero_()
                self.ee_static_set.E.zero_()
                wp.launch(cone_cone_collision_set_static, (self.n_edges, self.n_edges_static), inputs = [self.medial_geo, self.medial_geo_static, self.ee_static_set])
                ret += self.ee_static_set.E.numpy()[0]
        return ret


    def sphere(self, e0):
        ve0 = self.V[e0]

        r0 = self.R[e0]
        v_rst_0 = self.V_rest[e0]   

        s0 = MedialSphere(ve0, v_rst_0, r0, e0)
        return s0

    def analyze(self):

        b = np.zeros(self.n_vertices * 3)
        # if self.dense:
        #     H = np.zeros((self.n_vertices * 3, self.n_vertices * 3))

        rows = []
        cols = []
        blocks = []
        self.indices_set.clear()

        nss = self.pt_set.cnt.numpy()[0]
        ss_id = self.pt_set.a.numpy()[:nss]
        ncc = self.ee_set.cnt.numpy()[0]
        cc_id = self.ee_set.a.numpy()[:ncc]

        if self.ground is not None and sg_collision:
            n_ground = self.g_set.cnt.numpy()[0]
            self.sg_id = self.g_set.a.numpy()[:n_ground]
            self.sg_hr = self.g_set.hr.numpy()[:n_ground]

        self.cc_set = []
        self.cc_id = []
        
        self.ss_set = []
        self.ss_id = []
        
        n_repeated_cc = 0
        n_repeated_ss = 0
        for id in ss_id:
            # if tuple(id) in self.rest_pt:
            #     n_repeated_ss += 1
            #     continue
            e0, e1, e2, e3 = id
            s0, s1, s2, s3 = self.sphere(e0), self.sphere(e1), self.sphere(e2), self.sphere(e3)
            
            cons = SlabSphereConstraint(s0, s1, s2, s3)
            dist = cons.compute_distance()

            self.ss_set.append(cons)
            self.ss_id.append(id)

            
        for id in cc_id:
            # if tuple(id) in self.rest_cc:
            #     n_repeated_cc += 1
            #     continue
            e0, e1, e2, e3 = id

            if self.is_1_ring(e0, e1, e2, e3) or self.is_2_ring(e0, e1, e2, e3):
                continue

            s0, s1, s2, s3 = self.sphere(e0), self.sphere(e1), self.sphere(e2), self.sphere(e3)

            cons = ConeConeConstraint(s0, s1, s2, s3)
            dist = cons.compute_distance()

            self.cc_set.append(cons)
            self.cc_id.append(id)

        if COLLISION_DEBUG:
            print(f"filtered cc = {n_repeated_cc}, ss = {n_repeated_ss}")
            if len(self.cc_set):
                print(f"{len(self.cc_set)} collision detected")

        for ss, ssid in zip(self.ss_set, self.ss_id):
            e0, e1, e2, e3 = ssid

            E = [e0, e1, e2, e3]
            ee = np.array(E)# * 3

            dist = np.abs(ss.get_distance())
            g, h = ss.get_dist_gh()
            b[e0 * 3: (e0 + 1) * 3] += 2 * dist * g[:3]
            b[e1 * 3: (e1 + 1) * 3] += 2 * dist * g[3:6]
            b[e2 * 3: (e2 + 1) * 3] += 2 * dist * g[6:9]
            b[e3 * 3: (e3 + 1) * 3] += 2 * dist * g[9:12]
            
            h = 2 * dist * h + 2 * np.outer(g, g)

            self.indices_set.update(ee)
            # self.indices_set.update(ee + 1)
            # self.indices_set.update(ee + 2)
            # if self.dense:
            if False:
                for ii in range(4):
                    for jj in range(4):
                        H[E[ii] * 3: (E[ii] + 1) * 3, E[jj] * 3: (E[jj] + 1) * 3] += h[ii * 3: (ii + 1) * 3, jj * 3: (jj + 1) * 3]
            else:
                # ijl = [(ii, jj) for ii in range(3) for jj in range(3)] + [(ii, jj) for ii in range(3, 4) for jj in range(3, 4)]
                ijl = [(ii, jj) for ii in range(3) for jj in range(4)]

                for (ii, jj) in ijl:
                    rows.append(E[ii])
                    cols.append(E[jj])
                    blocks.append(h[ii * 3: (ii + 1) * 3, jj * 3: (jj + 1) * 3])

        for cc, ccid in zip(self.cc_set, self.cc_id):
            # i, j = ccid

            # e0, e1 = self.E[i]
            # e2, e3 = self.E[j] 
            e0, e1, e2, e3 = ccid

            E = [e0, e1, e2, e3]
            ee = np.array(E)# * 3

            dist = np.abs(cc.get_distance())
            g, h = cc.get_dist_gh()
            b[e0 * 3: (e0 + 1) * 3] += 2 * dist * g[:3]
            b[e1 * 3: (e1 + 1) * 3] += 2 * dist * g[3:6]
            b[e2 * 3: (e2 + 1) * 3] += 2 * dist * g[6:9]
            b[e3 * 3: (e3 + 1) * 3] += 2 * dist * g[9:12]
            
            h = 2 * dist * h + 2 * np.outer(g, g)

            self.indices_set.update(ee)
            # self.indices_set.update(ee + 1)
            # self.indices_set.update(ee + 2)
            # if self.dense:
            if False:
                for ii in range(4):
                    for jj in range(4):
                        H[E[ii] * 3: (E[ii] + 1) * 3, E[jj] * 3: (E[jj] + 1) * 3] += h[ii * 3: (ii + 1) * 3, jj * 3: (jj + 1) * 3]
            else:
                ijl = [(ii, jj) for ii in range(4) for jj in range(4)]
                # ijl = [(ii, jj) for ii in range(2) for jj in range(2)] + [(ii, jj) for ii in range(2, 4) for jj in range(2, 4)]
                for (ii, jj) in ijl:
                    rows.append(E[ii])
                    cols.append(E[jj])
                    blocks.append(h[ii * 3: (ii + 1) * 3, jj * 3: (jj + 1) * 3])

        if self.ground is not None and sg_collision:
            
            for id, di in zip(self.sg_id, self.sg_hr):
                h = di[0]
                r = di[1]
                hh = np.zeros((3, 3))
                # h11 = 4 * (3 * h * h - r * r)
                h11 = 12 * (h - r) ** 2
                hh[1, 1] = h11

                # gg1 = 4 * h * (h * h - r * r)
                gg1 = 4 * (h - r) ** 3
                gg = np.array([0.0, gg1, 0.0])

                b[id * 3: id * 3 + 3] += gg * ground_rel_stiffness
                rows.append(id)
                cols.append(id)
                blocks.append(hh * ground_rel_stiffness)
                
            self.indices_set.update(self.sg_id)

        if hasattr(self, "static_soup"):
            n_static = self.p_static_set.cnt.numpy()[0]
            static_id = self.p_static_set.a.numpy()[:n_static]
            self.n_static = n_static

            
            for id in static_id:
                i, t = id
                it0 = self.static_indices[t * 3 + 0]
                it1 = self.static_indices[t * 3 + 1]
                it2 = self.static_indices[t * 3 + 2]
                
                t0 = self.static_V[it0]
                t1 = self.static_V[it1]
                t2 = self.static_V[it2]
                tn = plane_normal(t0, t1, t2)


                h = np.dot(tn, self.V[i] - t0)
                r = self.R[i]

                hh = np.zeros((3, 3))
                # h11 = 4 * (3 * h * h - r * r)
                h11 = 12 * (h - r) ** 2
                # hh[1, 1] = h11
                hh += np.outer(tn, tn) * h11

                # gg1 = 4 * h * (h * h - r * r)
                gg1 = 4 * (h - r) ** 3
                # gg = np.array([0.0, gg1, 0.0])
                gg = gg1 * tn

                b[i * 3: i * 3 + 3] += gg * ground_rel_stiffness
                rows.append(i)
                cols.append(i)
                blocks.append(hh * ground_rel_stiffness)
            
            self.indices_set.update(static_id[:, 0])
            if self.static_objects.has_medials:
                n_static_ee = self.ee_static_set.cnt.numpy()[0]
                ee_static_id = self.ee_static_set.a.numpy()[:n_static_ee]
                self.n_static_ee = n_static_ee
                cc_static_set = []
                cc_static_id = []
                for id in ee_static_id:
                    e0, e1, e2, e3 = id


                    s0, s1 = self.sphere(e0), self.sphere(e1)
                    s2, s3 = self.static_objects.sphere(e2), self.static_objects.sphere(e3)

                    cons = ConeConeConstraint(s0, s1, s2, s3)
                    dist = cons.compute_distance()

                    cc_static_set.append(cons)
                    cc_static_id.append(id)
                    
                for cc, ccid in zip(cc_static_set, cc_static_id):
                    # i, j = ccid

                    # e0, e1 = self.E[i]
                    # e2, e3 = self.E[j] 
                    e0, e1, e2, e3 = ccid


                    E = [e0, e1]
                    ee = np.array(E)# * 3

                    dist = np.abs(cc.get_distance())
                    g, h = cc.get_dist_gh()
                    b[e0 * 3: (e0 + 1) * 3] += 2 * dist * g[:3]
                    b[e1 * 3: (e1 + 1) * 3] += 2 * dist * g[3:6]
                    # b[e2 * 3: (e2 + 1) * 3] += 2 * dist * g[6:9]
                    # b[e3 * 3: (e3 + 1) * 3] += 2 * dist * g[9:12]
                    
                    h = 2 * dist * h + 2 * np.outer(g, g)

                    self.indices_set.update(ee)
                    # self.indices_set.update(ee + 1)
                    # self.indices_set.update(ee + 2)
                    # if self.dense:
                    if False:
                        for ii in range(4):
                            for jj in range(4):
                                H[E[ii] * 3: (E[ii] + 1) * 3, E[jj] * 3: (E[jj] + 1) * 3] += h[ii * 3: (ii + 1) * 3, jj * 3: (jj + 1) * 3]
                    else:
                        for ii in range(2):
                            for jj in range(2):
                                rows.append(E[ii])
                                cols.append(E[jj])
                                blocks.append(h[ii * 3: (ii + 1) * 3, jj * 3: (jj + 1) * 3])


        if not self.dense:
            hh = bsr_zeros(self.n_vertices, self.n_vertices, wp.mat33, device = "cpu")
            bsr_set_from_triplets(hh, wp.array(rows, dtype = int, device = "cpu"), wp.array(cols, dtype = int, device = "cpu"), wp.array(blocks, dtype = wp.mat33, device= "cpu"))
            H = bsr_matrix((hh.values.numpy(), hh.columns.numpy(), hh.offsets.numpy()), shape = hh.shape, blocksize=(3, 3))
            # H = bsr_array((blocks, (rows, cols)), shape = (self.n_vertices * 3, self.n_vertices * 3), blocksize=(3, 3))
        idx = sorted(self.indices_set)
        H_dim = len(idx) * 3
        idx_inv = dict(zip(idx, range(len(idx))))
        H = np.zeros((H_dim, H_dim))
        for r, c, bb in zip(rows, cols, blocks):
            i = idx_inv[r]
            j = idx_inv[c]
            H[i * 3: (i + 1) * 3, j *3 : (j  +1) * 3] += bb
        
        idx = np.array(idx, int).reshape((-1, 1))
        ret_idx = np.hstack([idx * 3, idx * 3 + 1, idx * 3 + 2]).reshape(-1)
        return b[ret_idx], H, ret_idx

    def energy(self, V, R = None):
        return self.collision_set(V, R, energy_only = True)

        
    
        