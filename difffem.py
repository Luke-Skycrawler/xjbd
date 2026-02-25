import warp as wp
from stretch import RodBC, PSViewer, Triplets, should_fix, set_K_fixed, set_b_fixed
import polyscope as ps
from fem.linear_elasticity import PK1
from diff_utils import *
from warp.sparse import bsr_zeros, BsrMatrix, bsr_set_from_triplets, bsr_mv
from fem.fem import compute_Dm
from scipy.sparse import diags, identity, bsr_matrix, csr_matrix, csc_matrix, vstack, hstack
from scipy.sparse.linalg import splu
from warp.optim.linear import cg, bicgstab
import igl 
from geometry.static_scene import StaticScene
import polyscope.imgui as gui
h = 1e-2
h_fd = 1e-4

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

@wp.kernel
def f_kernel(x: wp.array(dtype = wp.vec3), x_target: wp.array(dtype = wp.vec3), f: wp.array(dtype = float)):
    '''
    loss function of differentiable simulation
    f = \sum wi / 2 * ||x - x_target||^2
    '''
    i = wp.tid()
    fi = wp.dot(x[i] - x_target[i], x[i] - x_target[i]) / 2.0

    wp.atomic_add(f, 0, fi)

@wp.kernel
def pfpx_kernel(x: wp.array(dtype = wp.vec3), x_target: wp.array(dtype = wp.vec3), pfpx: wp.array(dtype = wp.vec3)):
    '''
    partial f / partial x = wi * (x - x_target)
    treating wi = 1 for all nodes for now
    '''
    i = wp.tid()
    pfpx[i] = x[i] - x_target[i]


class DiffRodBC(RodBC):
    def __init__(self, h): 
        super().__init__(h) 
        self.pcpx_triplets = self.make_triplets()
        self.pcpp_triplets = self.make_triplets()
        self.pcpx_sparse = bsr_zeros(self.n_nodes, self.n_nodes, wp.mat33)
        self.pcpp_sparse = bsr_zeros(self.n_nodes, self.n_nodes, wp.mat33)
        target = wp.zeros_like(self.states.x)
        wp.copy(target, self.xcs)
        self.x_target = target 

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
        wp.launch(set_K_fixed, (self.n_tets * 4 * 4,), inputs = [self.geo, self.pcpp_triplets])

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
        x_xcs0 = self.equilibrium()

        np.random.seed(43)
        dpnp, dpwp = self._perturb_rest()
        
        pcpp_dp = wp.zeros_like(self.b)
        dxwp = wp.zeros_like(pcpp_dp)


        bsr_mv(self.pcpp_sparse, dpwp, pcpp_dp, transpose = True)
        self.set_fixed(pcpp_dp)

        cg(self.pcpx_sparse, pcpp_dp, dxwp, tol = 1e-10)
        # bicgstab(self.pcpx_sparse, pcpp_dp, dxwp, tol = 1e-10)

        self.set_fixed(dxwp)
        
        
        # self._debug_cg(dxwp, pcpp_dp)

        dx_predict = -dxwp.numpy()   
        
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
        
        self.compute_Dm_keep_W()
        # self.compute_Dm()
        self.compute_K()        
        bp = self.b.numpy()
        return bp

    def _verify_pcpp(self):
        '''
        fixme: not considering fixed dofs
        '''
        self.step()
        self.compute_pcpp()
        # self.pcpp_sparse = bsr_zeros(self.n_nodes, self.n_nodes, wp.mat33)
        bsr_set_from_triplets(self.pcpp_sparse, self.pcpp_triplets.rows, self.pcpp_triplets.cols, self.pcpp_triplets.vals)

        self.compute_K()

        # manipulate dx 
        dxnp = np.random.rand(self.n_nodes, 3) * h_fd
        dxwp = wp.array(dxnp, dtype = wp.vec3)
        # x_curr = self.states.x.numpy()
        x_curr = self.xcs.numpy()
        b_curr = self.b.numpy()

        bpdp = self._b_from_xc(x_curr, dxnp * 0.5)
        bmdp = self._b_from_xc(x_curr, dxnp * -0.5)
        db_predict_wp = wp.zeros_like(self.b)
        bsr_mv(self.pcpp_sparse, dxwp, db_predict_wp, transpose = True)
        
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
        
    def compute_dfdp(self, x_target): 
        '''
        Eq. (4) in [1]
        df/dp = (partial f/partial p) + partial f/partial x * S
        '''
        pfpx = wp.zeros_like(self.states.x)
        wp.launch(pfpx_kernel, (self.n_nodes, ), inputs = [self.states.x, x_target, pfpx])
        
        tmp = wp.zeros_like(pfpx)
        cg(self.pcpx_sparse, pfpx, tmp, tol = 1e-10)

        dfdp = wp.zeros_like(pfpx)
        bsr_mv(self.pcpp_sparse, tmp, dfdp, alpha = -1.0)
        return dfdp
    
    def compute_f(self, target): 
        f_curr = wp.zeros((1,), dtype = float)
        wp.launch(f_kernel, (self.n_nodes,), inputs = [self.states.x, target, f_curr])
        return f_curr.numpy()[0]

    def equilibrium(self):
        '''
        computes p c/ px and p c/ pp at equilibrium state
        '''
        self.step()
        # static equilibrium state that satisfies c(x, p) = 0
        x_xcs0 = self.states.x.numpy()
        igl.write_obj("x_xcs0.obj", x_xcs0, self.F)

        self.compute_pcpx()
        bsr_set_from_triplets(self.pcpx_sparse, self.pcpx_triplets.rows, self.pcpx_triplets.cols, self.pcpx_triplets.vals)

        self.compute_pcpp()
        bsr_set_from_triplets(self.pcpp_sparse, self.pcpp_triplets.rows, self.pcpp_triplets.cols, self.pcpp_triplets.vals)
        return x_xcs0


    def _perturb_rest(self, dp = None): 
        # helper function to verify the correctness of pcpp and pxpp with refernce finite difference jacobian

        if dp is None:
            np.random.seed(41)
            dpnp = np.random.rand(self.n_nodes, 3) * h_fd
        else: 
            dpnp = dp 

        dpwp = wp.array(dpnp, dtype = wp.vec3)
        self.set_fixed(dpwp)
        dpnp = dpwp.numpy()

        # set xcs to x + dp
        x_curr = self.xcs.numpy()
        xpdx = x_curr + dpnp
        self.xcs.assign(xpdx)

        self.compute_Dm_keep_W()
        # self.compute_Dm()
        self.reset()
        return dpnp, dpwp
        
    def _verify_dfdp(self):

        target = self.x_target
        self.equilibrium()
        f_curr = self.compute_f(target)
        dfdp = self.compute_dfdp(target)

        dpnp, dpwp = self._perturb_rest()

        df_predict = np.dot(dfdp.numpy().reshape(-1), dpnp.reshape(-1))

        self.step()
        # new equilibrium after perturbation
        x_xcs1 = self.states.x.numpy()
        igl.write_obj("x_xcs1.obj", x_xcs1, self.F) 

        f_new = self.compute_f(target)

        df = f_new - f_curr
        err = df - df_predict

        print("df: ", df)
        print("df predict: ", df_predict)
        print("df error: ", np.max(np.abs(err)))

    def assemble_sgn_hessian(self): 
        '''
        |   A           B^T         (p c / p x)^T   | |dx       | = |0              |
        |   B           C           (p c / p p)^T   | |dp       |   |-df / dp^T     |
        | p c / p x     p c / p p   0               | |d lambda |   |-p f / p x^T   |
        '''
        # A = sum wi (partial c/partial x) ^T (partial c/partial x)
        # 3n * 3n identity 
        A = identity(self.n_nodes * 3, dtype = float, format = "csc")
        B = csc_matrix((self.n_nodes * 3, self.n_nodes * 3), dtype = float)
        C = csc_matrix((self.n_nodes * 3, self.n_nodes * 3), dtype = float)

        pcpx = self.to_scipy_bsr(self.pcpx_sparse)
        pcpp = self.to_scipy_bsr(self.pcpp_sparse)

        sgn = vstack([
            hstack([A, B.T, pcpx]),
            hstack([B, C, pcpp]),
            hstack([pcpx.T, pcpp.T, B])], format = "csc")
        
        return sgn
        
    def assemble_sgn_rhs(self): 
        dfdp = self.compute_dfdp(self.x_target).numpy().reshape(-1)
        z3n = np.zeros_like(dfdp)
        rhs = np.concatenate([z3n, -dfdp, z3n])
        return rhs

    def optimize_rest_shape(self): 
        self.equilibrium()
        
        sgn = self.assemble_sgn_hessian()
        rhs = self.assemble_sgn_rhs()
        lu = splu(sgn)
        sol = lu.solve(rhs)
        dx = sol[:self.n_nodes * 3].reshape((-1, 3))
        dp = sol[self.n_nodes * 3: self.n_nodes * 6].reshape((-1, 3))

        self.states.x.assign(self.states.x.numpy() + dx)
        self.xcs.assign(self.xcs.numpy() + dp)
        
        self.compute_Dm_keep_W()
        
        return dp, dx

class RestShapeViewer(PSViewer): 
    def __init__(self, rod, static_mesh: StaticScene = None): 
        super().__init__(rod, static_mesh)
        
        self.ps_restshape = ps.register_surface_mesh("rest shape", self.V, self.F)

    def callback(self):
        super().callback()
        if gui.Button("optimize rest shape"):
            dp, dx = self.rod.optimize_rest_shape()
            print("dp: ", np.linalg.norm(dp))
            print("dx: ", np.linalg.norm(dx))

            V_rest = self.rod.xcs.numpy()
            self.ps_restshape.update_vertex_positions(V_rest)
        

        

def drape():
    ps.init()
    wp.init()
    # rod = RodBC(h, "assets/elephant.mesh")
    # rod = RodBC(h)
    rod = DiffRodBC(h)
    # rod._verify_pcpx()
    # rod._verify_pcpp()
    # rod._verify_dxdp()
    # rod._verify_dfdp()

    viewer = RestShapeViewer(rod)
    ps.set_user_callback(viewer.callback)
    ps.show()

    
if __name__ == "__main__":
    drape() 
