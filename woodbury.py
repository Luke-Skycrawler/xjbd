import numpy as np
from scipy.linalg import solve, lu_factor, lu_solve, inv
import warp as wp

class Woodbury:
    '''
    solve for (A + UCV) x = b using Woodbury matrix identity
    (A + UCV)^-1 = A^-1 - A^-1 U(C^-1 + VA^-1 U)^-1 VA^-1
    '''
    def __init__(self, A):
        self.A = A
        self.lu, self.piv = lu_factor(A)

    def apply_inv_A(self, b):
        x = lu_solve((self.lu, self.piv), b)
        return x

    def update(self, U, C, V):
        self.U = U
        self.C = C
        self.V = V
        self.A_inv_U = self.apply_inv_A(self.U)
        self.central_term = inv(self.C) + self.V @ self.A_inv_U
    def solve(self, b):
        vi = self.apply_inv_A(b)
        VA_inv = self.V @ vi

        # A_inv_U = self.apply_inv_A(self.U)
        # central_term = inv(self.C) + self.V @ A_inv_U
        
        term1 = vi

        # print(f"VA_inv shape = {VA_inv.shape}, b shape = {b.shape}, central term shape = {self.central_term.shape}")
        tmp = solve(self.central_term, VA_inv)
        # print(f"tmp = {tmp}")
        term2 = self.A_inv_U @ tmp

        return term1 - term2


def test():
    
    n, k = 2400, 16 * 3
    A = np.random.ranf((n, n))
    b = np.random.rand((n))
    
    A = A + A.T
    
    U = np.random.ranf((n, k))
    V = np.random.ranf((k, n))
    C = np.random.ranf((k, k))
    C = C + C.T

    wb = Woodbury(A)
    wb.update(U, C, V)
    
    with wp.ScopedTimer("woodbury solve"):
        for _ in range(20):
            x0 = wb.solve(b)

    with wp.ScopedTimer("direct solve"):
        for _ in range(20):
            x1 = solve(A +  U @ C @ V, b)
    print(f"diff = {np.linalg.norm(x0 - x1)}, x0 norm = {np.linalg.norm(x0)}, x1 norm = {np.linalg.norm(x1)}")

if __name__ == "__main__":
    test()