import numpy as np 
from scipy.spatial.transform import Rotation as R
def vec(t):
    return (t.T).reshape(-1)

def vec_inv(v):
    return v.reshape((3, -1)).T 

def to_world(q): 
    qA = q[:12]
    qV = q[12:]
    
    A = vec_inv(qA[:9])
    I4 = np.identity(4, dtype = np.float32)
    qV_world = np.kron(I4, A) @ qV
    return np.concatenate([qA, qV_world])
    
def fd_test():
    magnitude = 1e-1
    qV = np.random.rand(12) * magnitude 
    # generate random rotation matrix 
    qA3 = vec(R.from_euler('xyz', np.random.rand(3) * 360, degrees=True).as_matrix())
    qAb = np.random.rand(3)
    qA = np.concatenate([qA3, qAb])
    A = vec_inv(qA[:9])
    I4 = np.identity(4, dtype = np.float32)

    q = np.concatenate([qA, qV])
    
    dq = np.random.rand(24)
    h = 1e-3
    q_world_minus = to_world(q - dq * h)
    q_world_plus = to_world(q + dq * h)
    
    dq_world_fd = (q_world_plus - q_world_minus) / (2)

    dqt_dq = np.zeros((24, 24))
    dqt_dq[:12, :12] = np.identity(12)
    dqt_dq[12:, 12:] = np.kron(I4, A)
    

    for ii in range(3):
        for jj in range(4):
            eij = np.zeros((3, 3))
            if jj < 3:
                eij[ii, jj] = 1.0
            row = ii + jj * 3
            column = ii + jj * 3 + 12
            vij = np.kron(I4, eij) @ qV
            dqt_dq[row, 12:] = vij.reshape((1, -1))
            # dqt_dq[:12, column] = vij.reshape((1, -1))
    
    dq_world_pred = dqt_dq @ dq * h

    diff = dq_world_fd - dq_world_pred
    
    print(f"norm dq_world_fd = {np.linalg.norm(dq_world_fd):.2e}, norm dq_world_pred = {np.linalg.norm(dq_world_pred):.2e}, norm diff = {np.linalg.norm(diff):.2e}")

if __name__ == "__main__":
    fd_test()
    
    