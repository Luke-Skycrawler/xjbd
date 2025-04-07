import warp as wp
import numpy as np
from geometry.collision_cell import EdgeEdgeCollisionList, CollisionList, GroundCollisionList
from mipctk import MedialSphere, ConeConeConstraint, SlabSphereConstraint
CC_SET_SIZE = 256
SS_SET_SIZE = 256

# @wp.kernel
# def collision_medial(edges: wp.array(dtype = wp.vec2i), vertices: wp.array(dtype = wp.vec3), radius: wp.array(dtype = float), ee_set: EdgeEdgeCollisionList):
#     i, j = wp.tid()
#     if i < j: 
#         # brute force edge-edge collision detection
        
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
        self.vertices = wp.array(V_medial, dtype = wp.vec3)
        self.radius = wp.array(R, dtype = float)
        self.edges = wp.array(E, dtype = wp.vec2i)
        self.faces = wp.array(F, dtype = wp.vec3i)

        self.n_vertices = V_medial.shape[0]
        self.n_edges = E.shape[0]
        self.n_faces = F.shape[0]

        self.ee_set = EdgeEdgeCollisionList()
        self.ee_set.a = wp.zeros(CC_SET_SIZE, dtype = wp.vec2i)
        self.ee_set.cnt = wp.zeros(1, dtype = int)

        
        
    def refit(self, V, R = None):
        if R is not None:
            self.R[:] = R
            self.radius.assign(R)
        self.V[:] = V
        self.vertices.assign(V)

    def collision_set(self, V, R = None):
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
            for i in range(n_edges - 1, n_edges):
                # FIXME: ad-hoc for now, only detect collision with the last edge
                for j in range(i):
                    # brute force edge-edge collision detection
                    e0, e1 = self.E[i]
                    e2, e3 = self.E[j] 

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
            i, j = ccid

            e0, e1 = self.E[i]
            e2, e3 = self.E[j] 
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

    
                    

            
            
        
        
    
        