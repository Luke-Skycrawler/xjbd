import igl.triangle
import warp as wp 
import numpy as np
from fem.params import FEMMesh
from fem.fem import Triplets
from typing import List
import polyscope as ps
import igl
from warp.sparse import *

wp.config.max_unroll = 0
wp.init()
from collision.hl import *
from collision.ee import C_ee, dceedx_s, dcdx_delta_ee
from collision.vf import C_vf, dcvfdx_s, dcdx_delta_vf
from collision.dcdx_delta import *

COLLISION_DEBUG = False
collision_eps = 2e-2
PT_SET_SIZE = 4096
EE_SET_SIZE = 4096
GROUND_SET_SIZE = 4096
FLT_MAX = 1e5
ZERO = 1e-6
stiffness = 1e5

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

@wp.struct
class GroundCollisionList:
    a: wp.array(dtype = int)
    cnt: wp.array(dtype = int)

@wp.func
def append(cl: GroundCollisionList, element: int):
    id = wp.atomic_add(cl.cnt, 0, 1)
    cl.a[id] = element

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
        # n_triangles = triangles_soup.indices.shape[0] // 3
        # for y in range(n_triangles):
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


@wp.func
def within_2_ring(i: int, j: int, edges: wp.array(dtype = int)) -> bool:
    return False

@wp.kernel
def point_plane_collision(triangle_soup: TriangleSoup, ground_plane: float, collision_list: GroundCollisionList):
    i = wp.tid()
    xi = triangle_soup.vertices[i]
    if xi[1] < ground_plane:
        append(collision_list, i)

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

    inside_2_ring = within_2_ring(x, closest, edges) 
    if closest != -1 and not inside_2_ring and closest_distance < collision_eps:
        collision = wp.vec2i(x, closest)
        append(collision_list, collision, bary_closest)

@wp.kernel
def edge_aabbs(edges: wp.array(dtype = int), vertices: wp.array(dtype = wp.vec3), lowers: wp.array(dtype = wp.vec3), uppers: wp.array(dtype = wp.vec3)):
    i = wp.tid()
    e0 = vertices[edges[i * 2 + 0]]
    e1 = vertices[edges[i * 2 + 1]]
    lowers[i] = wp.min(e0, e1)  
    uppers[i] = wp.max(e0, e1)


@wp.func
def psd(x: float) -> float:
    return wp.sqrt(wp.max(0.0, x))

@wp.kernel
def fill_collision_triplets(pt_set: CollisionList, triangle_soup: TriangleSoup, triplets: Triplets, rhs: wp.array(dtype = wp.vec3), stiffness: float):
    tid = wp.tid()
    pt = pt_set.a[tid]
    x = pt[0]
    t = pt[1]

    it0 = triangle_soup.indices[t * 3]
    it1 = triangle_soup.indices[t * 3 + 1]
    it2 = triangle_soup.indices[t * 3 + 2]


    idx = wp.vec4i(x, it0, it1, it2)

    x0 = triangle_soup.vertices[x]
    x1 = triangle_soup.vertices[it0]
    x2 = triangle_soup.vertices[it1]
    x3 = triangle_soup.vertices[it2]


    e0p, e1p, e2p = C_vf(x0, x1, x2, x3)
    # dcdx_delta_vf(x0, x1, x2, x3, dcdx_delta)
    ds = dcvfdx_s(x0, x1, x2, x3)
    l = signed_distance(e0p, e1p, e2p)

    if l > 0.0:
        pass
    else: 
        gl0, gl1, gl2 = gl(l, e2p)
        lam0, lam1, lam2, lam3, q0, q1, q2, q3 = eig_Hl(e0p, e1p, e2p)
        lam4 = 2.0
        q4 = wp.mat33(0.0)
        q4[2] = gl2
        lam_diag = lam_tilde_diag(q0, q1, q2, q3, q4, lam0, lam1, lam2, lam3, lam4)

        # set forces
        for i in range(4):
            id = idx[i]
            gi = 2.0 * e2p * ds[2, i] * stiffness
            # gi = gl2 * ds[2, i] * stiffness
            # as gl0 = gl1 = 0
            wp.atomic_add(rhs, id, -gi)


        
        Lam = wp.diag(lam_diag)
        dcdx_simple_mat = wp.matrix(
            ds[0, 0], 0., 0.,     ds[0, 1], 0., 0.,     ds[0, 2], 0., 0.,     ds[0, 3], 0., 0.,
            0., ds[0, 0], 0.,     0., ds[0, 1], 0.,     0., ds[0, 2], 0.,     0., ds[0, 3], 0.,
            0., 0., ds[0, 0],     0., 0., ds[0, 1],     0., 0., ds[0, 2],     0., 0., ds[0, 3],

            ds[1, 0], 0., 0.,     ds[1, 1], 0., 0.,     ds[1, 2], 0., 0.,     ds[1, 3], 0., 0.,
            0., ds[1, 0], 0.,     0., ds[1, 1], 0.,     0., ds[1, 2], 0.,     0., ds[1, 3], 0.,
            0., 0., ds[1, 0],     0., 0., ds[1, 1],     0., 0., ds[1, 2],     0., 0., ds[1, 3],

            ds[2, 0], 0., 0.,     ds[2, 1], 0., 0.,     ds[2, 2], 0., 0.,     ds[2, 3], 0., 0.,
            0., ds[2, 0], 0.,     0., ds[2, 1], 0.,     0., ds[2, 2], 0.,     0., ds[2, 3], 0.,
            0., 0., ds[2, 0],     0., 0., ds[2, 1],     0., 0., ds[2, 2],     0., 0., ds[2, 3],

            shape = (9, 12)
        )
        q_mat = wp.matrix(
            q0[0, 0], q1[0, 0], q2[0, 0], q3[0, 0], q4[0, 0],
            q0[0, 1], q1[0, 1], q2[0, 1], q3[0, 1], q4[0, 1],
            q0[0, 2], q1[0, 2], q2[0, 2], q3[0, 2], q4[0, 2],
            q0[1, 0], q1[1, 0], q2[1, 0], q3[1, 0], q4[1, 0],
            q0[1, 1], q1[1, 1], q2[1, 1], q3[1, 1], q4[1, 1],
            q0[1, 2], q1[1, 2], q2[1, 2], q3[1, 2], q4[1, 2],
            q0[2, 0], q1[2, 0], q2[2, 0], q3[2, 0], q4[2, 0],
            q0[2, 1], q1[2, 1], q2[2, 1], q3[2, 1], q4[2, 1],
            q0[2, 2], q1[2, 2], q2[2, 2], q3[2, 2], q4[2, 2],
            shape = (9, 5)
        )

        K = wp.transpose(dcdx_simple_mat) @ q_mat @ Lam
        d2Psidx2 = K @ wp.transpose(K)
        for ii in range(4):
            for jj in range(4):
                i = idx[ii]
                j = idx[jj]
                triplet_id = tid * 16  + 4 * ii +  jj
                triplets.rows[triplet_id] = i
                triplets.cols[triplet_id] = j
                triplets.vals[triplet_id] = wp.mat33(
                    d2Psidx2[ii * 3, jj * 3], d2Psidx2[ii * 3, jj * 3 + 1], d2Psidx2[ii * 3, jj * 3 + 2],
                    d2Psidx2[ii * 3 + 1, jj * 3], d2Psidx2[ii * 3 + 1, jj * 3 + 1], d2Psidx2[ii * 3 + 1, jj * 3 + 2],
                    d2Psidx2[ii * 3 + 2, jj * 3], d2Psidx2[ii * 3 + 2, jj * 3 + 1], d2Psidx2[ii * 3 + 2, jj * 3 + 2]
                ) * stiffness

@wp.func
def triangle_edge_intersection(t0: wp.vec3, t1: wp.vec3, t2: wp.vec3, e0: wp.vec3, e1: wp.vec3):
    ret = False
    origin = e0
    dir = wp.normalize(e1 - e0)
    
    diff = t0 - origin 
    normal = plane_normal(t0, t1, t2)
    denom = wp.dot(normal, dir)
    if wp.abs(denom) < ZERO:
        ret = False
    else:
        t = wp.dot(diff , normal) / denom
        if t < 0.0:
            ret = False
        else: 
            h = origin + t * dir
            test1 = wp.dot(normal, wp.cross(t1 - t0, h - t0)) < 0.0
            test2 = wp.dot(normal, wp.cross(t2 - t1, h - t1)) < 0.0
            test3 = wp.dot(normal, wp.cross(t0 - t2, h - t2)) < 0.0

            if test1 or test2 or test3:
                ret = False
            elif t < wp.length(e1 - e0):
                ret = True
    return ret

@wp.func
def fetch_triangle_points(i: int, x: wp.array(dtype = wp.vec3), indices: wp.array(dtype = int)):
    ia = indices[i * 3]
    ib = indices[i * 3 + 1]
    ic = indices[i * 3 + 2]
    return x[ia], x[ib], x[ic]

@wp.kernel
def fill_collision_triplets_ground(triplets_offset: int, ground_plane: float, ground_set: GroundCollisionList, triangle_soup: TriangleSoup, triplets: Triplets, rhs: wp.array(dtype = wp.vec3), stiffness: float):
    tid = wp.tid()
    i = ground_set.a[tid]
    xi = triangle_soup.vertices[i]

    triplet_idx = triplets_offset * 16 + tid
    if xi[1] < ground_plane:
        gi = -2.0 * wp.vec3(0.0, ground_plane - xi[1], 0.0) * stiffness
        wp.atomic_add(rhs, i, -gi)
        hi = wp.mat33(0.0)
        hi[1, 1] = 2.0 * stiffness
        triplets.rows[triplet_idx] = i
        triplets.cols[triplet_idx] = i
        triplets.vals[triplet_idx] = hi

@wp.kernel
def fill_collision_triplets_ee(npt: int, ee_set: EdgeEdgeCollisionList, triangle_soup: TriangleSoup, triplets: Triplets, rhs: wp.array(dtype = wp.vec3), stiffness: float, edges: wp.array(dtype = int), neighbor_faces: wp.array(dtype = int)):

    tid = wp.tid()
    ee = ee_set.a[tid]
    x = ee[0]
    y = ee[1]

    iea0 = edges[x * 2]
    iea1 = edges[x * 2 + 1]
    ieb0 = edges[y * 2]
    ieb1 = edges[y * 2 + 1]

    x0 = triangle_soup.vertices[iea0]
    x1 = triangle_soup.vertices[iea1]
    x2 = triangle_soup.vertices[ieb0]  
    x3 = triangle_soup.vertices[ieb1]  

    idx = wp.vec4i(iea0, iea1, ieb0, ieb1)
    e0p, e1p, e2p = C_ee(x0, x1, x2, x3)
    ds = dceedx_s(x0, x1, x2, x3)
    l = signed_distance(e0p, e1p, e2p)

    e0 = x0 
    e1 = x1
    ia = neighbor_faces[y * 2]
    ib = neighbor_faces[y * 2 + 1]
    
    ta0, ta1, ta2 = fetch_triangle_points(ia, triangle_soup.vertices, triangle_soup.indices)
    tb0, tb1, tb2 = fetch_triangle_points(ib, triangle_soup.vertices, triangle_soup.indices)

    is_penetrating = triangle_edge_intersection(ta0, ta1, ta2, e0, e1) or triangle_edge_intersection(tb0, tb1, tb2, e0, e1)
    
    if is_penetrating: 
        gl0, gl1, gl2 = gl(l, e2p)
        lam0, lam1, lam2, lam3, q0, q1, q2, q3 = eig_Hl(e0p, e1p, e2p)
        lam4 = 2.0
        q4 = wp.mat33(0.0)
        q4[2] = gl2

        lam_diag = lam_tilde_diag(q0, q1, q2, q3, q4, lam0, lam1, lam2, lam3, lam4)
        
        # set_forces
        for i in range(4):
            id = idx[i]
            gi = 2.0 * e2p * ds[2, i] * stiffness
            wp.atomic_add(rhs, id, -gi)

        Lam = wp.diag(lam_diag)
        dcdx_simple_mat = wp.matrix(
            ds[0, 0], 0., 0.,     ds[0, 1], 0., 0.,     ds[0, 2], 0., 0.,     ds[0, 3], 0., 0.,
            0., ds[0, 0], 0.,     0., ds[0, 1], 0.,     0., ds[0, 2], 0.,     0., ds[0, 3], 0.,
            0., 0., ds[0, 0],     0., 0., ds[0, 1],     0., 0., ds[0, 2],     0., 0., ds[0, 3],

            ds[1, 0], 0., 0.,     ds[1, 1], 0., 0.,     ds[1, 2], 0., 0.,     ds[1, 3], 0., 0.,
            0., ds[1, 0], 0.,     0., ds[1, 1], 0.,     0., ds[1, 2], 0.,     0., ds[1, 3], 0.,
            0., 0., ds[1, 0],     0., 0., ds[1, 1],     0., 0., ds[1, 2],     0., 0., ds[1, 3],

            ds[2, 0], 0., 0.,     ds[2, 1], 0., 0.,     ds[2, 2], 0., 0.,     ds[2, 3], 0., 0.,
            0., ds[2, 0], 0.,     0., ds[2, 1], 0.,     0., ds[2, 2], 0.,     0., ds[2, 3], 0.,
            0., 0., ds[2, 0],     0., 0., ds[2, 1],     0., 0., ds[2, 2],     0., 0., ds[2, 3],

            shape = (9, 12)
        )
        q_mat = wp.matrix(
            q0[0, 0], q1[0, 0], q2[0, 0], q3[0, 0], q4[0, 0],
            q0[0, 1], q1[0, 1], q2[0, 1], q3[0, 1], q4[0, 1],
            q0[0, 2], q1[0, 2], q2[0, 2], q3[0, 2], q4[0, 2],
            q0[1, 0], q1[1, 0], q2[1, 0], q3[1, 0], q4[1, 0],
            q0[1, 1], q1[1, 1], q2[1, 1], q3[1, 1], q4[1, 1],
            q0[1, 2], q1[1, 2], q2[1, 2], q3[1, 2], q4[1, 2],
            q0[2, 0], q1[2, 0], q2[2, 0], q3[2, 0], q4[2, 0],
            q0[2, 1], q1[2, 1], q2[2, 1], q3[2, 1], q4[2, 1],
            q0[2, 2], q1[2, 2], q2[2, 2], q3[2, 2], q4[2, 2],
            shape = (9, 5)
        )

        K = wp.transpose(dcdx_simple_mat) @ q_mat @ Lam
        d2Psidx2 = K @ wp.transpose(K)

        for ii in range(4):
            for jj in range(4):
                i = idx[ii]
                j = idx[jj]
                triplet_id = tid * 16  + 4 * ii +  jj + npt * 16
                triplets.rows[triplet_id] = i
                triplets.cols[triplet_id] = j
                triplets.vals[triplet_id] = wp.mat33(
                    d2Psidx2[ii * 3, jj * 3], d2Psidx2[ii * 3, jj * 3 + 1], d2Psidx2[ii * 3, jj * 3 + 2],
                    d2Psidx2[ii * 3 + 1, jj * 3], d2Psidx2[ii * 3 + 1, jj * 3 + 1], d2Psidx2[ii * 3 + 1, jj * 3 + 2],
                    d2Psidx2[ii * 3 + 2, jj * 3], d2Psidx2[ii * 3 + 2, jj * 3 + 1], d2Psidx2[ii * 3 + 2, jj * 3 + 2]
                ) * stiffness
        
@wp.func
def lam_tilde_diag(q0: wp.mat33, q1: wp.mat33, q2: wp.mat33, q3: wp.mat33, q4: wp.mat33, lam0: float, lam1: float, lam2: float, lam3: float, lam4: float):
    q0Tq0 = wp.length_sq(q0[0]) + wp.length_sq(q0[1]) + wp.length_sq(q0[2])
    q1Tq1 = wp.length_sq(q1[0]) + wp.length_sq(q1[1]) + wp.length_sq(q1[2])
    q2Tq2 = wp.length_sq(q2[0]) + wp.length_sq(q2[1]) + wp.length_sq(q2[2])
    q3Tq3 = wp.length_sq(q3[0]) + wp.length_sq(q3[1]) + wp.length_sq(q3[2])
    q4Tq4 = wp.length_sq(q4[0]) + wp.length_sq(q4[1]) + wp.length_sq(q4[2])

    lam_tilde_0 = lam0 / q0Tq0
    lam_tilde_1 = lam1 / q1Tq1
    lam_tilde_2 = lam2 / q2Tq2
    lam_tilde_3 = lam3 / q3Tq3
    lam_tilde_4 = lam4 / q4Tq4


    lam_diag = wp.vector(
        psd(lam_tilde_0),
        psd(lam_tilde_1),
        psd(lam_tilde_2),
        psd(lam_tilde_3),
        psd(lam_tilde_4)
    )
    return lam_diag

@wp.kernel
def collision_energy_ground(ground_plane: float, g_set: GroundCollisionList, triangle_soup: TriangleSoup, stiffness: float, e_col: wp.array(dtype = float)):
    tid = wp.tid()
    i = g_set.a[tid]
    x = triangle_soup.vertices[i]
    if x[1] < ground_plane:
        dh = ground_plane - x[1]
        dpsi = dh * dh * stiffness
        wp.atomic_add(e_col, 0, dpsi)
    
@wp.kernel
def collision_energy_pt(pt_set: CollisionList, triangle_soup: TriangleSoup, inverted: wp.array(dtype = int), stiffness: float, e_col: wp.array(dtype = float)):
    tid = wp.tid()
    pt = pt_set.a[tid]
    x = pt[0]
    t = pt[1]

    it0 = triangle_soup.indices[t * 3]
    it1 = triangle_soup.indices[t * 3 + 1]
    it2 = triangle_soup.indices[t * 3 + 2]


    idx = wp.vec4i(x, it0, it1, it2)

    x0 = triangle_soup.vertices[x]
    x1 = triangle_soup.vertices[it0]
    x2 = triangle_soup.vertices[it1]
    x3 = triangle_soup.vertices[it2]


    e0p, e1p, e2p = C_vf(x0, x1, x2, x3)
    # dcdx_delta_vf(x0, x1, x2, x3, dcdx_delta)
    # ds = dcvfdx_s(x0, x1, x2, x3)
    l = signed_distance(e0p, e1p, e2p)

    if l > 0.0: 
        pass
    else:
        dpsi = l * l * stiffness
        wp.atomic_add(e_col, 0, dpsi)

@wp.kernel
def collision_energy_ee(ee_set: EdgeEdgeCollisionList, triangle_soup: TriangleSoup, inverted: wp.array(dtype = int), stiffness: float, edges: wp.array(dtype = int), neighbor_faces: wp.array(dtype = int), e_col: wp.array(dtype = float)):
    tid = wp.tid()
    ee = ee_set.a[tid]
    x = ee[0]
    y = ee[1]

    iea0 = edges[x * 2]
    iea1 = edges[x * 2 + 1]
    ieb0 = edges[y * 2]
    ieb1 = edges[y * 2 + 1]

    x0 = triangle_soup.vertices[iea0]
    x1 = triangle_soup.vertices[iea1]
    x2 = triangle_soup.vertices[ieb0]  
    x3 = triangle_soup.vertices[ieb1]  

    e0p, e1p, e2p = C_ee(x0, x1, x2, x3)
    l = signed_distance(e0p, e1p, e2p)

    e0 = x0 
    e1 = x1
    ia = neighbor_faces[y * 2]
    ib = neighbor_faces[y * 2 + 1]
    
    ta0, ta1, ta2 = fetch_triangle_points(ia, triangle_soup.vertices, triangle_soup.indices)
    tb0, tb1, tb2 = fetch_triangle_points(ib, triangle_soup.vertices, triangle_soup.indices)

    is_penetrating = triangle_edge_intersection(ta0, ta1, ta2, e0, e1) or triangle_edge_intersection(tb0, tb1, tb2, e0, e1)

    if is_penetrating: 
        dpsi = l * l * stiffness
        wp.atomic_add(e_col, 0, dpsi)
    
    
@wp.kernel
def disable_interior_vertices(triangle_soup: TriangleSoup, inverted: wp.array(dtype = int)):
    i = wp.tid()
    idx =  triangle_soup.indices[i]
    inverted[idx] = 0

class MeshCollisionDetector:
    def __init__(self, xcs, T, indices, Bm, ground = None):
        '''
        heavy-lifting class for collision detection provided the reference to warp array of points and indices
        '''
        self.xcs = xcs
        self.T = T
        self.indices = indices
        self.Bm = Bm
        self.n_nodes = n_nodes = xcs.shape[0]
        if T is None:
            T = wp.array((0, 4), dtype = int)
        self.n_tets = n_tets = T.shape[0]
        self.n_triangles = n_triangles = indices.shape[0] // 3
        
        self.ground_enabled = True
        self.ground = 0.0
        if ground is None:
            self.ground_enabled = False
            self.ground = None
        else:
            self.ground = ground

        triangle_soup = TriangleSoup()
        triangle_soup.indices = indices
        triangle_soup.vertices = xcs
        
        self.mesh_triangle_soup = wp.Mesh(triangle_soup.vertices, triangle_soup.indices)
        triangle_soup.mesh_id = self.mesh_triangle_soup.id
        self.triangle_soup = triangle_soup

        self.F = indices.numpy().reshape((-1, 3))
        
        self.E = igl.edges(self.F).reshape(-1)
        self.n_edges = self.E.shape[0] // 2
        # self.edges = wp.array(self.E, dtype = int)
        self.edges = wp.zeros((self.n_edges * 2, ), dtype = int)
        self.edges.assign(self.E)


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

        ground_set = GroundCollisionList()
        ground_set.cnt = wp.zeros(shape = (1,), dtype = int)
        ground_set.a = wp.zeros(shape = (GROUND_SET_SIZE, ), dtype = int)

        inverted = wp.zeros(shape = (n_nodes,), dtype = int)

        self.inverted = inverted
        self.pt_set = pt_set
        self.ee_set = ee_set
        self.ground_set = ground_set

        self.e_col = wp.zeros((1, ), dtype = float)

        # FIXME: only here for a test. change it to actual neighing data 
        TT, _ = igl.triangle_triangle_adjacency(self.F)
        _, _, EF = igl.edge_topology(np.zeros((self.n_nodes, 3), dtype = float), self.F)
        self.neighbor_faces = wp.array(EF.reshape(-1), dtype = int)
        self.neighbors = wp.zeros((n_triangles * 3, ), dtype = int)
        print(f"TT min = {np.min(TT)}")
        self.neighbors.assign(TT.reshape(-1))
        # self.neighbors.assign(np.array([-1] * n_triangles * 3, dtype = int))
        # FIXME: might have no neighbors if it is a cloth
        self.stiffness = stiffness

    def compute_edge_aabbs(self):
        wp.launch(edge_aabbs, dim = (self.n_edges,), inputs = [self.edges, self.triangle_soup.vertices, self.lowers, self.uppers])

    def refit(self):
        self.compute_edge_aabbs()
        self.edges_bvh.refit()
        self.mesh_triangle_soup.refit()

    def collision_set(self, type = "all"):
        with wp.ScopedTimer("refit"):
            self.refit()
        npt, nee = 0, 0
        n_ground = 0

        with wp.ScopedTimer("ground"):
            if self.ground_enabled and (type == "ground" or type == "all"):
                self.ground_set.cnt.zero_()
                wp.launch(point_plane_collision, dim = (self.n_nodes, ), inputs = [self.triangle_soup, self.ground, self.ground_set])
                # n_ground = self.ground_set.cnt.numpy()[0]
                if COLLISION_DEBUG:
                    min_y = np.min(self.triangle_soup.vertices.numpy()[:, 1])
                    print(f"verts y coord min = {min_y}, ground = {self.ground}")
        with wp.ScopedTimer("ee"):
            if type == "ee" or type == "all":
                self.ee_set.cnt.zero_()
                wp.launch(edge_edge_collison, dim = (self.n_edges,), inputs = [self.edges, self.triangle_soup, self.edges_bvh.id, self.ee_set])
                # nee = self.ee_set.cnt.numpy()[0]
                # col_set = self.ee_set.a.numpy()[:nee]
                # np.save("ee_set.npy", col_set)
                # np.save("edges.npy", self.edges.numpy())
                # np.save("points.npy", self.triangle_soup.vertices.numpy())
        with wp.ScopedTimer("pt"):
            if type == "pt" or type == "all":
                self.pt_set.cnt.zero_()

                if self.Bm is not None:
                    self.inverted.fill_(1)
                    wp.launch(disable_interior_vertices, dim = (self.n_triangles * 3,), inputs = [self.triangle_soup, self.inverted])
                    wp.launch(compute_inverted_vertex, dim = (self.n_tets,), inputs = [self.triangle_soup.vertices, self.T, self.Bm, self.inverted])
                    if COLLISION_DEBUG:
                        print(f"inverted sum = {np.sum(self.inverted.numpy())}")
                # self.inverted.zero_()

                wp.launch(point_triangle_collision, dim = (self.n_nodes, ), inputs = [self.inverted, self.triangle_soup, self.neighbors, self.pt_set])
                # npt = self.pt_set.cnt.numpy()[0]

        npt, nee, n_ground = self.pt_set.cnt.numpy()[0], self.ee_set.cnt.numpy()[0], self.ground_set.cnt.numpy()[0]
        return npt, nee, n_ground

    def collision_energy(self, npt = 0, nee = 0, n_ground = 0):
        self.e_col.zero_()
        wp.launch(collision_energy_pt, (npt, ), inputs = [self.pt_set, self.triangle_soup, self.inverted, self.stiffness, self.e_col])

        wp.launch(collision_energy_ee, (nee, ), inputs = [self.ee_set, self.triangle_soup, self.inverted, self.stiffness, self.edges, self.neighbor_faces, self.e_col])

        wp.launch(collision_energy_ground, (n_ground,), inputs = [self.ground, self.ground_set, self.triangle_soup, self.stiffness, self.e_col])

        ret = self.e_col.numpy()[0]
        if COLLISION_DEBUG:
            print(F"npt = {npt}, nee = {nee}, n_ground = {n_ground}, collision energy = {ret}")
        return ret
        # return 0.0

    def analyze(self, rhs, npt = 0, nee = 0, n_ground = 0):
        '''
        returns the force and force derivatives
        '''
        triplets = Triplets()
        triplets.rows = wp.zeros(((npt + nee) * 16 + n_ground, ), dtype = int)
        triplets.cols = wp.zeros_like(triplets.rows)
        triplets.vals = wp.zeros(((npt + nee) * 16 + n_ground, ), dtype = wp.mat33)
        wp.launch(fill_collision_triplets, (npt, ), inputs = [self.pt_set, self.triangle_soup, triplets, rhs, self.stiffness])
        

        # triplets_ee = Triplets()
        # triplets_ee.rows = wp.zeros((nee * 16, ), dtype = int)
        # triplets_ee.cols = wp.zeros_like(triplets_ee.rows)
        # triplets_ee.vals = wp.zeros((nee * 16, ), dtype = wp.mat33)

        wp.launch(fill_collision_triplets_ee, (nee, ), inputs = [npt, self.ee_set, self.triangle_soup, triplets, rhs, self.stiffness, self.edges, self.neighbor_faces])
        # wp.copy(triplets.rows, triplets_ee.rows, npt * 16, 0, nee * 16)
        # wp.copy(triplets.cols, triplets_ee.cols, npt * 16, 0, nee * 16)
        # wp.copy(triplets.vals, triplets_ee.vals, npt * 16, 0, nee * 16)

        wp.launch(fill_collision_triplets_ground, (n_ground, ), inputs = [npt + nee, self.ground, self.ground_set, self.triangle_soup, triplets, rhs, self.stiffness])
        return triplets

class TestViewer:
    '''
    naming conventions:
    numpy arrays are capitalized
    warp arrays are lower case
    '''
    def __init__(self):
        
        n_triangles = 2
        n_nodes = 6

        rng = np.random.default_rng(123)
        rand_verts = rng.random(size = (n_nodes, 3), dtype = float)
        # fix the first triangle to x-z plane
        rand_verts[:3, 1] = 0.0

        self.indices = wp.array(np.arange(n_triangles * 3), dtype = int)
        self.vertices = wp.zeros((n_nodes, ), dtype = wp.vec3)
        self.vertices.assign(rand_verts)

        self.collider = collider = MeshCollisionDetector(self.vertices, None, self.indices, None)


        self.V = collider.triangle_soup.vertices.numpy()
        self.V0 = np.copy(self.V[:n_triangles * 3, :])
        self.F = collider.triangle_soup.indices.numpy().reshape((-1, 3))

        self.ps_mesh = ps.register_surface_mesh("triangle", self.V0[:3, ], self.F[:1])
        self.ps_mesh_fixed = ps.register_surface_mesh("triangle_fixed", self.V0[3:, ], self.F[1:] - 3)

        self.ps_mesh.set_transform_gizmo_enabled(True)

        # ps.set_user_callback(self.callback)
        ps.set_user_callback(self.callback_ee)

        self.n_nodes, self.n_triangles = n_nodes, n_triangles
        self.ps_points = ps.register_point_cloud("points", self.V)
        self.ps_points.set_radius(collision_eps)
        ps.show()

    def move_geometry(self):
        trans = self.ps_mesh.get_transform()
        # only controls the first triangle here
        V0 = np.zeros((3, 4))
        V0[:, :3] = self.V0[: 3]
        V0[:, 3] = 1.0
        V = (V0 @ trans.T)[:, : 3]
        self.V[:3, :] = V

        # should all mean the same thing and upating one is equivalent to updating all 
        self.collider.mesh_triangle_soup.points.assign(self.V)
        self.collider.xcs.assign(self.V)

        self.ps_points.update_point_positions(self.V)
    


    def callback_ee(self):
        self.move_geometry()

        # get updated uppers and lowers for edge aabbs  
        self.collider.compute_edge_aabbs()
        self.collider.edges_bvh.refit()

        _, nee, _ = self.collider.collision_set("ee")
        
        if nee:
            self.ps_mesh.set_color((1.0, 0.0, 0.0))
        else: 
            self.ps_mesh.set_color((0.0, 0.0, 0.0))    

    def callback(self):
        self.move_geometry()

        self.collider.mesh_triangle_soup.refit()
        
        npt, _, _ = self.collider.collision_set("pt")
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

    