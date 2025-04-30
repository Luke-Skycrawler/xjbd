import warp as wp
import numpy as np 
L, W = 1, 0.2
# mu, rho, lam = 2e6, 1., 125.0
mu, rho, lam = 2e6, 1.0e3, 0.0
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
gravity = wp.vec3(0, -g, 0)
gravity_np = np.array([0.0, -g, 0])

# damping
alpha = 6
beta = 1e-7
dt = 1e-3

model = "bar2"
default_tobj = f"assets/{model}.tobj"
# default_tobj = "bunny_5.tobj"

@wp.struct
class FEMMesh:
    n_nodes: int
    n_tets: int 
    xcs: wp.array(dtype = wp.vec3)
    T: wp.array2d(dtype = int)
    indices: wp.array(dtype = int)
