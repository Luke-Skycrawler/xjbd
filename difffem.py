import warp as wp
from stretch import RodBC, PSViewer, Triplets, should_fix, set_K_fixed, set_b_fixed
import polyscope as ps
from fem.linear_elasticity import PK1
from diff_utils import *
from warp.sparse import bsr_zeros, BsrMatrix, bsr_set_from_triplets, bsr_mv
from fem.fem import compute_Dm
from scipy.sparse.linalg import splu
from warp.optim.linear import cg, bicgstab
import igl 
h = 1e-2
h_fd = 1e-3

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
    e = ej // 4
    _i = ej % 4
    
    t0 = x[geo.T[e, 0]]
    t1 = x[geo.T[e, 1]]
    t2 = x[geo.T[e, 2]]
    t3 = x[geo.T[e, 3]]
    
    Ds = wp.mat33(t0 - t3, t1 - t3, t2 - t3)
    
    F = Ds @ Bm[e]
    i = geo.T[e, _i]
    
    dHdx0i = dHdx0(F, Bm[e], _i) * W[e]
    # 9x3 matrix 
    cols = wp.transpose(dHdx0i)
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

    

class DiffRodBC(RodBC):
    def __init__(self, h): 
        super().__init__(h) 
        self.pcpx_triplets = self.make_triplets()
        self.pcpp_triplets = self.make_triplets()
        self.pcpx_sparse = bsr_zeros(self.n_nodes, self.n_nodes, wp.mat33)
        self.pcpp_sparse = bsr_zeros(self.n_nodes, self.n_nodes, wp.mat33)

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
        wp.launch(set_K_fixed, (self.n_tets * 4 * 4,), inputs = [self.geo, self.pcpx_triplets])

    def compute_pcpp(self,):
        self.pcpp_triplets.rows.zero_()
        self.pcpp_triplets.cols.zero_()
        self.pcpp_triplets.vals.zero_()
        wp.launch(pcpp_sparse, (self.n_tets * 4,), inputs = [self.states.x, self.geo, self.Bm, self.W, self.pcpp_triplets])
        # wp.launch(set_K_fixed, (self.n_tets * 4 * 4,), inputs = [self.geo, self.pcpp_triplets])

    def set_fixed(self, wp_array): 
        wp.launch(set_b_fixed, (self.n_nodes,), inputs = [self.geo, wp_array])

    def _debug_cg(self, dxwp, pcpp_dp):
        pcpx_dx = wp.zeros_like(dxwp)
        bsr_mv(self.pcpx_sparse, dxwp, pcpx_dx)
        err_cg = pcpx_dx.numpy() - pcpp_dp.numpy()
        print("cg solver error: ", np.max(np.abs(err_cg)))
        print("pcpp dp: ", pcpp_dp.numpy(), "pcpx dp: ", pcpx_dx.numpy())
        
    def _verify_dxdp(self):
        '''
        Eq. (3) in [1]
        sensitivity matrix S = dx / dp = -(partial c / partial x)^{-1} @ (partial c / partial p)
        '''
        self.step()
        # static equilibrium state that satisfies c(x, p) = 0
        x_xcs0 = self.states.x.numpy()
        igl.write_obj("x_xcs0.obj", x_xcs0, self.F)

        self.compute_pcpx()
        bsr_set_from_triplets(self.pcpx_sparse, self.pcpx_triplets.rows, self.pcpx_triplets.cols, self.pcpx_triplets.vals)

        self.compute_pcpp()
        bsr_set_from_triplets(self.pcpp_sparse, self.pcpp_triplets.rows, self.pcpp_triplets.cols, self.pcpp_triplets.vals)

        np.random.seed(43)
        dpnp = np.random.rand(self.n_nodes, 3) * h_fd
        dpwp = wp.array(dpnp, dtype = wp.vec3)
        self.set_fixed(dpwp)
        dpnp = dpwp.numpy()
        
        pcpp_dp = wp.zeros_like(self.b)
        dxwp = wp.zeros_like(pcpp_dp)

        bsr_mv(self.pcpp_sparse, dpwp, pcpp_dp)
        self.set_fixed(pcpp_dp)

        cg(self.pcpx_sparse, pcpp_dp, dxwp, tol = 1e-10)
        # bicgstab(self.pcpx_sparse, pcpp_dp, dxwp, tol = 1e-10)

        self.set_fixed(dxwp)
        
        
        # self._debug_cg(dxwp, pcpp_dp)

        dx_predict = -dxwp.numpy()   
        
        # set xcs to x + dp
        x_curr = self.xcs.numpy()
        xpdx = x_curr + dpnp
        self.xcs.assign(xpdx)

        self.compute_Dm_keep_W()
        # self.compute_Dm()
        self.reset()
        self.step()
        # new equilibrium after perturbation
        x_xcs1 = self.states.x.numpy()
        igl.write_obj("x_xcs1.obj", x_xcs1, self.F) 

        dx = x_xcs1 - x_xcs0
        err = dx - dx_predict
        print("dx: ", dx)
        print("dx predict: ", dx_predict)
        print("dx error: ", np.max(np.abs(err)))
        
    # def set_bc_fixed_hessian(self):
    #     pass

    # def set_bc_fixed_grad(self):
    #     pass

    def compute_Dm_keep_W(self):
        W = wp.zeros_like(self.W)
        wp.launch(compute_Dm, (self.n_tets, ), inputs = [self.geo, self.Bm, W])

    def _b_from_xc(self, x_curr, dxnp):
        xpdx = x_curr + dxnp
        self.xcs.assign(xpdx)
        
        # self.compute_Dm_keep_W()
        self.compute_Dm()
        self.compute_K()        
        bp = self.b.numpy()
        return bp

    def _verify_pcpp(self):
        '''
        fixme: not considering fixed dofs
        '''
        self.compute_pcpp()
        # self.pcpp_sparse = bsr_zeros(self.n_nodes, self.n_nodes, wp.mat33)
        bsr_set_from_triplets(self.pcpp_sparse, self.pcpp_triplets.rows, self.pcpp_triplets.cols, self.pcpp_triplets.vals)

        self.tet_kernel_sparse()

        # manipulate dx 
        dxnp = np.random.rand(self.n_nodes, 3) * h_fd
        dxwp = wp.array(dxnp, dtype = wp.vec3)
        # x_curr = self.states.x.numpy()
        x_curr = self.xcs.numpy()
        b_curr = self.b.numpy()

        bpdp = self._b_from_xc(x_curr, dxnp * 0.5)
        bmdp = self._b_from_xc(x_curr, dxnp * -0.5)
        db_predict_wp = wp.zeros_like(self.b)
        bsr_mv(self.pcpp_sparse, dxwp, db_predict_wp)
        
        db = bpdp - bmdp 
        db_predict = db_predict_wp.numpy()
        err = db + db_predict
        print("db: ", db)
        print("db predict: ", db_predict)

        print("db error: ", np.max(np.abs(err)))

    def _verify_pcpx(self):
        self.step()
        # start from drapped position; optionally can also start from rest position

        self.compute_pcpx()
        # self.pcpx_sparse = bsr_zeros(self.n_nodes, self.n_nodes, wp.mat33)
        bsr_set_from_triplets(self.pcpx_sparse, self.pcpx_triplets.rows, self.pcpx_triplets.cols, self.pcpx_triplets.vals)

        diff_r = self.pcpx_triplets.rows.numpy() - self.triplets.rows.numpy()
        diff_c = self.pcpx_triplets.cols.numpy() - self.triplets.cols.numpy()
        print("row diff: ", np.max(np.abs(diff_r)))
        print("col diff: ", np.max(np.abs(diff_c)))

        # should be same as stiffness matrix 
        self.compute_K()
        ref = self.triplets.vals.numpy()
        test = self.pcpx_triplets.vals.numpy()
        err = np.linalg.norm(ref - test, axis = (1, 2))
        print("pcpx error: ", np.max(np.abs(err)))


        # manipulate dx 
        dxnp = np.random.rand(self.n_nodes, 3) * h_fd
        dxwp = wp.array(dxnp, dtype = wp.vec3)
        self.set_fixed(dxwp)
        dxnp = dxwp.numpy()

        x_curr = self.states.x.numpy()
        x_dx = x_curr + dxnp
        self.states.x.assign(x_dx)

        self.set_bc_fixed_grad()
        b_curr = self.b.numpy()

        db_predict_wp = wp.zeros_like(self.b)
        bsr_mv(self.pcpx_sparse, dxwp, db_predict_wp)
        

        self.compute_K()        
        self.set_bc_fixed_grad()
        db = self.b.numpy() - b_curr 
        db_predict = db_predict_wp.numpy()
        err = db + db_predict
        # print("db: ", db)
        # print("db predict: ", db_predict)

        print("db error: ", np.max(np.abs(err)))
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
    # rod._verify_pcpx()
    rod._verify_pcpp()

    # viewer = PSViewer(rod)
    # ps.set_user_callback(viewer.callback)
    # ps.show()
    
if __name__ == "__main__":
    drape() 
