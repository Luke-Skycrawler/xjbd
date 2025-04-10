import warp as wp
import numpy as np
from geometry.collision_cell import EdgeEdgeCollisionList, CollisionList, GroundCollisionList
from mipctk import MedialSphere, ConeConeConstraint, SlabSphereConstraint

from g2m.analyze import compute_distance_cone_cone
CC_SET_SIZE = 256
SS_SET_SIZE = 256

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
    

@wp.struct
class ConeConeCollisionList:
    a: wp.array(dtype = wp.vec4i)
    cnt: wp.array(dtype = int)

@wp.struct
class SlabSphereCollisionList:
    a: wp.array(dtype = wp.vec4i)
    cnt: wp.array(dtype = int)


@wp.func
def append(cc_list: ConeConeCollisionList, element: wp.vec4i):
    id = wp.atomic_add(cc_list.cnt, 0, 1)
    cc_list.a[id] = element

@wp.kernel
def cone_cone_collision_set(geo: MedialGeometry, cc_list: ConeConeCollisionList):
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
        if dist < 0.0:
            col = wp.vec4i(ex[0], ex[1], ey[0], ey[1])
            append(cc_list, col)
    
class MedialCollisionDetector:
    def __init__(self, V_medial, R, E, F, ground = None): 
        '''
        medial-medial collision detection & response
        '''

        # numpy arrays
        self.V = V_medial
        self.V_rest = np.copy(V_medial)
        self.R = R
        self.E = E
        self.F = F

        self.cc_set = []
        self.ss_set = []

        
        # warp arrays 
        self.medial_geo = MedialGeometry()
        
        self.medial_geo.vertices = wp.array(V_medial, dtype = wp.vec3)
        self.medial_geo.radius = wp.array(R, dtype = float)
        self.medial_geo.edges = wp.array(E, dtype = wp.vec2i)
        self.medial_geo.faces = wp.array(F, dtype = wp.vec3i)

        self.n_vertices = V_medial.shape[0]
        self.n_edges = E.shape[0]
        self.n_faces = F.shape[0]

        self.ee_set = ConeConeCollisionList()
        self.ee_set.a = wp.zeros(CC_SET_SIZE, dtype = wp.vec4i)
        self.ee_set.cnt = wp.zeros(1, dtype = int)

        vv = lambda e: (max(e[0], e[1]), min(e[0], e[1])) 
        self.vv_adjacency = set([vv(e) for e in E])
        
        
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

    def collision_set(self, V, R = None):
        self.refit(V, R)
        nps, ncc = 0, 0

        n_ground = 0
        
        self.ee_set.cnt.zero_()
        wp.launch(cone_cone_collision_set, (self.n_edges, self.n_edges), inputs = [self.medial_geo, self.ee_set])
        ncc = self.ee_set.cnt.numpy()[0]
        cc_id = self.ee_set.a.numpy()[:ncc]

        self.cc_set = []
        self.cc_id = []
        for id in cc_id:
            e0, e1, e2, e3 = id

            if self.is_1_ring(e0, e1, e2, e3) or self.is_2_ring(e0, e1, e2, e3):
                continue

            ve0 = self.V[e0]
            ve1 = self.V[e1]
            ve2 = self.V[e2]
            ve3 = self.V[e3]

            r0 = self.R[e0]
            r1 = self.R[e1]
            r2 = self.R[e2]
            r3 = self.R[e3]

            v_rst_0 = self.V_rest[e0]   
            v_rst_1 = self.V_rest[e1]
            v_rst_2 = self.V_rest[e2]
            v_rst_3 = self.V_rest[e3]

            s0 = MedialSphere(ve0, v_rst_0, r0, e0)
            s1 = MedialSphere(ve1, v_rst_1, r1, e1)
            s2 = MedialSphere(ve2, v_rst_2, r2, e2)
            s3 = MedialSphere(ve3, v_rst_3, r3, e3)

            cons = ConeConeConstraint(s0, s1, s2, s3)
            dist = cons.compute_distance()

            self.cc_set.append(cons)
            self.cc_id.append(id)

        if len(self.cc_set):
            print(f"{len(self.cc_set)} collision detected")
    def collision_set_slow(self, V, R = None):
        self.refit(V, R)
        # point-slab, cone-cone
        nps, ncc = 0, 0

        n_ground = 0

        # with wp.ScopedTimer("ground"):
        #     pass

        n_vertices = self.n_vertices
        n_edges = self.n_edges

        self.cc_set = []
        self.ss_set = []

        self.cc_id = []
        self.ss_id = []

        with wp.ScopedTimer("cone-cone"):

            # wp.launch(collision_medial, (n_edges, n_edges), inputs = [self.edges, self.vertices, self.radius, self.ee_set])
            for i in range(n_edges):
                # FIXME: ad-hoc for now, only detect collision with the last edge
                for j in range(i):
                    # brute force edge-edge collision detection
                    e0, e1 = self.E[i]
                    e2, e3 = self.E[j] 

                    if self.is_1_ring(e0, e1, e2, e3) or self.is_2_ring(e0, e1, e2, e3):
                        continue

                    ve0 = self.V[e0]
                    ve1 = self.V[e1]
                    ve2 = self.V[e2]
                    ve3 = self.V[e3]

                    r0 = self.R[e0]
                    r1 = self.R[e1]
                    r2 = self.R[e2]
                    r3 = self.R[e3]

                    v_rst_0 = self.V_rest[e0]   
                    v_rst_1 = self.V_rest[e1]
                    v_rst_2 = self.V_rest[e2]
                    v_rst_3 = self.V_rest[e3]

                    s0 = MedialSphere(ve0, v_rst_0, r0, e0)
                    s1 = MedialSphere(ve1, v_rst_1, r1, e1)
                    s2 = MedialSphere(ve2, v_rst_2, r2, e2)
                    s3 = MedialSphere(ve3, v_rst_3, r3, e3)

                    cons = ConeConeConstraint(s0, s1, s2, s3)
                    dist = cons.compute_distance()
                    if dist < 0:
                        self.cc_set.append(cons)
                        self.cc_id.append((i, j))
                        ncc += 1
                        print(f"collision detected")

    def analyze(self):
        b = np.zeros(self.n_vertices * 3)
        H = np.zeros((self.n_vertices * 3, self.n_vertices * 3))
        for cc, ccid in zip(self.cc_set, self.cc_id):
            # i, j = ccid

            # e0, e1 = self.E[i]
            # e2, e3 = self.E[j] 
            e0, e1, e2, e3 = ccid

            if self.is_1_ring(e0, e1, e2, e3) or self.is_2_ring(e0, e1, e2, e3):
                continue

            E = [e0, e1, e2, e3]

            g, h = cc.get_dist_gh()
            b[e0 * 3: (e0 + 1) * 3] += g[:3]
            b[e1 * 3: (e1 + 1) * 3] += g[3:6]
            b[e2 * 3: (e2 + 1) * 3] += g[6:9]
            b[e3 * 3: (e3 + 1) * 3] += g[9:12]

            for ii in range(4):
                for jj in range(4):
                    H[E[ii] * 3: (E[ii] + 1) * 3, E[jj] * 3: (E[jj] + 1) * 3] += h[ii * 3: (ii + 1) * 3, jj * 3: (jj + 1) * 3]
    
        return b, H

    
                    

            
            
        
        
    
        