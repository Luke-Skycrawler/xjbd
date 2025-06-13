import warp as wp
import numpy as np 
from admm_pd import ADMM_PD, ADMMState
from geometry.collision_cell import MeshCollisionDetector, collision_eps
from stretch import PSViewer
import polyscope as ps
class LBFGS_PD(ADMM_PD):
    def __init__(self, meshes = ["assets/bar2.tobj"], transforms = [np.identity(4)], h = 1e-2):
        super().__init__(meshes, transforms, h)
        self.m = 4  # number of previous steps to store
        self.stashed_states = np.zeros((self.m, self.n_nodes * 3))
        self.stashed_gradients = np.zeros_like(self.stashed_states)
        self.n_iters = 10

    def update_x(self):
        '''
        Algorithm 2 in [2]
        '''
        q = -self.b
        xk = self.states.x.numpy().reshape(-1)
        gk = self.b

        self.stashed_states[self.iter % self.m] = xk
        self.stashed_gradients[self.iter % self.m] = gk

        start = max(0, self.iter - self.m)
        if self.iter == 0:
            super().update_x()
            return 

        for i in reversed(range(start, self.iter)):
            xip1 = self.stashed_states[(i + 1) % self.m]
            xi = self.stashed_states[i % self.m]
            si = xip1 - xi 
            
            gi = self.stashed_gradients[i % self.m]
            gip1 = self.stashed_gradients[(i + 1) % self.m]
            ti = gip1 - gi
            
            rhoi = np.dot(ti, si)
            zetai = np.dot(q, si) / rhoi
            q -= zetai * ti

        
        r = self.solver.solve(q)
        for i in range(start, self.iter):
            xip1 = self.stashed_states[(i + 1) % self.m]
            xi = self.stashed_states[i % self.m]
            si = xip1 - xi 
            
            gi = self.stashed_gradients[i % self.m]
            gip1 = self.stashed_gradients[(i + 1) % self.m]
            ti = gip1 - gi
            
            rhoi = np.dot(ti, si)
            zetai = np.dot(q, si) / rhoi

            eta = np.dot(ti, r) / rhoi
            r += si * (zetai - eta)

        self.states.dx.assign(-r)
        self.line_search()

def test_pd():
    rod = ADMM_PD()
    viewer = PSViewer(rod)
    ps.set_user_callback(viewer.callback)

def staggered_bars():
    n_meshes = 2 
    meshes = ["assets/bar2.tobj"] * n_meshes
    # meshes = ["assets/bunny_5.tobj"] * n_meshes
    transforms = [np.identity(4, dtype = float) for _ in range(n_meshes)]
    transforms[1][:3, :3] = np.zeros((3, 3))
    transforms[1][0, 1] = 1
    transforms[1][1, 0] = 1
    transforms[1][2, 2] = 1

    for i in range(n_meshes):
        # transforms[i][0, 3] = i * 0.5
        transforms[i][1, 3] = 1.2 + i * 0.2
        transforms[i][2, 3] = i * 1.0
    
    rods = LBFGS_PD(meshes, transforms)
    viewer = PSViewer(rods)
    ps.set_user_callback(viewer.callback)

if __name__ == "__main__":
    # vis_eigs()
    ps.init() 
    ps.set_ground_plane_mode("none")
    wp.config.max_unroll = 0
    wp.init()

    # test_pd()
    staggered_bars()
    ps.show()
