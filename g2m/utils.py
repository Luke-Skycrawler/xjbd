import numpy as np 
from scipy.spatial.transform import Rotation as R
n_modes = 10
def dqs_Q(sQ, comp = True):
    lam = np.load("data/lam_effel.npy")[1:11]
    Q = np.zeros(sQ.shape)
    Q[:] = sQ[:]
    # Q_max = np.max(np.abs(Q), axis = 0, keepdims = True)
    # Q /= Q_max
    
    Q_max_signed = np.max(Q, axis = 0, keepdims = True)
    Q_min = np.min(Q, axis = 0, keepdims = True)
    Q_range = Q_max_signed - Q_min
    Q -= Q_min
    # Q /= Q_range

    if not comp:
        return Q
    comp_Q = 1 - Q
    ret = np.hstack([Q, comp_Q])
    Q_range = lam / 3e7
    return ret, Q_range.reshape((1, -1)), Q_min.reshape((1, -1))

def euler_to_quat(euler, Q_range = None):
    q = np.copy(euler.reshape((-1, 3)))
    q[:, 0] /= 3.
    q[:, 2] /= 3.    
    quats = np.zeros((n_modes * 2, 4))
    quats[:, 3] = 1.0
    for i in range(n_modes):
        if Q_range is not None:
            qr = Q_range[0, i]
        else: 
            qr = 1.0
        rotation = R.from_euler('xyz', q[i] / qr, degrees = False)
        quats[i] = rotation.as_quat()
    return quats

def euler_to_affine(euler, Q_range = None):
    q = np.copy(euler.reshape((-1, 3)))
    q[:, 0] /= 3.
    q[:, 2] /= 3.    
    affines = np.zeros((n_modes * 2, 12))
    affines[:, : 9] = np.eye(3).reshape(-1)
    for i in range(n_modes):
        if Q_range is not None:
            qr = Q_range[0, i]
        else: 
            qr = 1.0
        rotation = R.from_euler('xyz', q[i] / qr, degrees = False)
        rr = rotation.as_matrix().T
        rrt = np.vstack([rr, np.zeros((1, 3))])
        affines[i] = rrt.reshape(-1)
    return affines.reshape(-1) / n_modes
    
def npy_to_dataset(q, Q_range = None):
    if q.shape[1] == 120: 
        q_120d = q
    elif q.shape[1] == 36:
        q_120d = np.zeros((q.shape[0], 40), np.float32)
        for i in range(q.shape[0]):
            q_120d[i] = euler_to_quat(q[i], Q_range)[:10, :].reshape(-1)
    return q_120d
