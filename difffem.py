import warp as wp
from stretch import RodBC, PSViewer
import polyscope as ps
h = 1e-2

'''
reference: 
[1]: SGN: Sparse Gauss-Newton for Accelerated Sensitivity Analysis
'''

def drape():
    ps.init()
    wp.init()
    # rod = RodBC(h, "assets/elephant.mesh")
    rod = RodBC(h)
    viewer = PSViewer(rod)
    ps.set_user_callback(viewer.callback)
    ps.show()


if __name__ == "__main__":
    drape() 


