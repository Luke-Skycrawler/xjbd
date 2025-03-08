import warp as wp
from stretch import RodBCBase, PSViewer, NewtonState
from fem.interface import Rod, RodComplex
import polyscope as ps
import numpy as np
# collision add-ons
from geometry.collision_cell import MeshCollisionDetector

h = 1e-2
        
class RodComplexBC(RodBCBase, RodComplex):
    def __init__(self, h, meshes = [], transforms = []):
        self.meshes_filename = meshes 
        self.transforms = transforms
        super().__init__(h)
        self.collider = MeshCollisionDetector(self.xcs, self.T, self.indices)

    def set_bc_fixed_hessian(self):
        pass

    def set_bc_fixed_grad(self):
        pass

    def step(self):
        newton_iter = True
        n_iter = 0
        max_iter = 5
        # while n_iter < max_iter:
        while newton_iter:
            self.compute_A()
            self.compute_rhs()

            n_pt = self.collider.collision_set("pt")

            
            self.solve()
            # wp.launch(add_dx, dim = (self.n_nodes, ), inputs = [self.states, 1.0])
            
            
            # line search stuff, not converged yet
            alpha = self.line_search()
            if alpha == 0.0:
                break

            dxnp = self.states.dx.numpy()
            norm_dx = np.linalg.norm(dxnp)
            newton_iter = norm_dx > 1e-3 and n_iter < max_iter
            print(f"norm = {np.linalg.norm(dxnp)}, {n_iter}")
            n_iter += 1
        self.update_x0_xdot()

    
def multiple_drape():
    n_meshes = 3
    meshes = ["assets/bar2.tobj"] * n_meshes
    transforms = [np.identity(4, dtype = float) for _ in range(n_meshes)]
    for i in range(n_meshes):
        transforms[i][2, 3] = i * 2.0

    rods = RodComplexBC(h, meshes, transforms)
    viewer = PSViewer(rods)
    ps.set_user_callback(viewer.callback)
    ps.show()


@wp.kernel
def set_velocity_kernel(states: NewtonState):
    i = wp.tid()
    states.xdot[i] = wp.vec3(0.0)
    if i >= 525:
        states.xdot[i] = wp.vec3(0.0, 0.0, -3.0)
    
        
def set_velocity(rods: RodComplexBC):
    wp.launch(set_velocity_kernel, (rods.n_nodes,), inputs = [rods.states])

def staggered_bars():
    n_meshes = 2 
    meshes = ["assets/bar2.tobj"] * n_meshes
    transforms = [np.identity(4, dtype = float) for _ in range(n_meshes)]
    for i in range(n_meshes):
        transforms[i][2, 3] = i * 1.0
        transforms[i][0, 3] = i * 0.5
    
    rods = RodComplexBC(h, meshes, transforms)
    set_velocity(rods)
    rods.states.x0
    viewer = PSViewer(rods)
    ps.set_user_callback(viewer.callback)
    ps.show()

if __name__ == "__main__":
    ps.init()
    wp.init()

    # multiple_drape()
    # drape()
    staggered_bars()