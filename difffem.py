import warp as wp
from stretch import RodBC, PSViewer, Triplets
import polyscope as ps
from fem.linear_elasticity import PK1
from diff_utils import *
from warp.sparse import bsr_zeros, BsrMatrix, bsr_set_from_triplets, bsr_mv
h = 1e-2

'''
reference: 
[1]: SGN: Sparse Gauss-Newton for Accelerated Sensitivity Analysis
'''

@wp.kernel
def pcpx_sparse(x: wp.array(dtype = wp.vec3), geo: FEMMesh, Bm: wp.array(dtype = wp.mat33), W: wp.array(dtype = float), triplets: Triplets):
    ej = wp.tid()
    e = ej // 4
    _i = ej % 4
    
    t0 = x[geo.T[e, 0]]
    t1 = x[geo.T[e, 1]]
    t2 = x[geo.T[e, 2]]
    t3 = x[geo.T[e, 3]]
    
    Ds = wp.mat33(t0 - t3, t1 - t3, t2 - t3)
    
    F = Ds @ Bm[e]
    i = geo.T[e, _i]
    
    dHdxi = dHdx(F, Bm[e], _i) * W[e]
    # 9x3 matrix 
    cols = wp.transpose(dHdxi)
    c0 = inv_vec(cols[0])
    c1 = inv_vec(cols[1])
    c2 = inv_vec(cols[2])

    df0dxi = wp.matrix_from_rows(c0[0], c1[0], c2[0])
    df1dxi = wp.matrix_from_rows(c0[1], c1[1], c2[1])
    df2dxi = wp.matrix_from_rows(c0[2], c1[2], c2[2])
    df3dxi = -df0dxi - df1dxi - df2dxi

    cnt = ej * 4
    for jj in range(4): 
        j = geo.T[e, jj]
        triplets.rows[cnt + jj] = i 
        triplets.cols[cnt + jj] = j 
    
    triplets.vals[cnt] = df0dxi 
    triplets.vals[cnt + 1] = df1dxi
    triplets.vals[cnt + 2] = df2dxi
    triplets.vals[cnt + 3] = df3dxi

@wp.kernel
def pcpp_sparse(x: wp.array(dtype = wp.vec3), geo: FEMMesh, Bm: wp.array(dtype = wp.mat33), W: wp.array(dtype = float), triplets: Triplets):
    ej = wp.tid()
    # e = ej // 4
    # _i = ej % 4
    
    # t0 = x[geo.T[e, 0]]
    # t1 = x[geo.T[e, 1]]
    # t2 = x[geo.T[e, 2]]
    # t3 = x[geo.T[e, 3]]
    
    # Ds = wp.mat33(t0 - t3, t1 - t3, t2 - t3)
    
    # F = Ds @ Bm[e]
    # i = geo.T[e, _i]
    
    # dHdxi = dHdx(F, Bm[e], _i)
    # # 9x3 matrix 
    # cols = wp.transpose(dHdxi)
    # # c0 = wp.transpose(inv_vec(cols[0]))
    # # c1 = wp.transpose(inv_vec(cols[1]))
    # # c2 = wp.transpose(inv_vec(cols[2]))
    # c0 = inv_vec(cols[0])
    # c1 = inv_vec(cols[1])
    # c2 = inv_vec(cols[2])


    # # df0dxi = wp.matrix_from_cols(c0[0], c1[0], c2[0])
    # # df1dxi = wp.matrix_from_cols(c0[1], c1[1], c2[1])
    # # df2dxi = wp.matrix_from_cols(c0[2], c1[2], c2[2])
    # df0dxi = wp.matrix_from_rows(c0[0], c1[0], c2[0])
    # df1dxi = wp.matrix_from_rows(c0[1], c1[1], c2[1])
    # df2dxi = wp.matrix_from_rows(c0[2], c1[2], c2[2])
    # df3dxi = -df0dxi - df1dxi - df2dxi

    # cnt = ej * 4
    # for jj in range(4): 
    #     j = geo.T[e, jj]
    #     triplets.rows[cnt + j] = i 
    #     triplets.cols[cnt + j] = j 
    
    # triplets.vals[cnt] = df0dxi 
    # triplets.vals[cnt + 1] = df1dxi
    # triplets.vals[cnt + 2] = df2dxi
    # triplets.vals[cnt + 3] = df3dxi
    

class DiffRodBC(RodBC):
    def __init__(self, h): 
        super().__init__(h) 
        self.pcpx_triplets = self.make_triplets()
        self.pcpp_triplets = self.make_triplets()

    def make_triplets(self):
        
        triplets = Triplets()
        triplets.rows = wp.zeros((self.n_tets * 16,), dtype = int)
        triplets.cols = wp.zeros_like(triplets.rows)
        triplets.vals = wp.zeros((self.n_tets * 16,), dtype = wp.mat33)

        return triplets

    def compute_pcpx(self):
        self.pcpx_triplets.rows.zero_()
        self.pcpx_triplets.cols.zero_()
        self.pcpx_triplets.vals.zero_()
        wp.launch(pcpx_sparse, (self.n_tets * 4,), inputs = [self.states.x, self.geo, self.Bm, self.W, self.pcpx_triplets])
        

    def compute_pcpp(self,):
        self.pcpp_triplets.rows.zero_()
        self.pcpp_triplets.cols.zero_()
        self.pcpp_triplets.vals.zero_()
        wp.launch(pcpp_sparse, (self.n_tets * 4,), inputs = [self.states.x, self.geo, self.Bm, self.W, self.pcpp_triplets])


    def _verify_pcpx(self):
        self.compute_pcpx()
        self.pcpx_sparse = bsr_zeros(self.n_nodes, self.n_nodes, wp.mat33)
        bsr_set_from_triplets(self.pcpx_sparse, self.pcpx_triplets.rows, self.pcpx_triplets.cols, self.pcpx_triplets.vals)

        diff_r = self.pcpx_triplets.rows.numpy() - self.triplets.rows.numpy()
        diff_c = self.pcpx_triplets.cols.numpy() - self.triplets.cols.numpy()
        print("row diff: ", np.max(np.abs(diff_r)))
        print("col diff: ", np.max(np.abs(diff_c)))

        # should be same as stiffness matrix 
        self.tet_kernel_sparse()
        ref = self.triplets.vals.numpy()
        test = self.pcpx_triplets.vals.numpy()
        err = np.linalg.norm(ref - test, axis = (1, 2))
        print("pcpx error: ", np.max(np.abs(err)))
        # for l, r in zip(ref[:3], test[:3]):
        #     print("ref: ", l)
        #     print("test: ", r)
        
        # dxnp = np.random.rand(self.n_nodes, 3)
        # dxwp = wp.array(dxnp, dtype = wp.vec3)
        # dcwp = wp.zeros_like(dxwp)
        # bsr_mv(self.pcpx_sparse, dxwp, dcwp)
        
        
def drape():
    ps.init()
    wp.init()
    # rod = RodBC(h, "assets/elephant.mesh")
    # rod = RodBC(h)
    rod = DiffRodBC(h)
    rod._verify_pcpx()

    # viewer = PSViewer(rod)
    # ps.set_user_callback(viewer.callback)
    # ps.show()
    


    
    

if __name__ == "__main__":
    drape() 


