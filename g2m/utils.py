import numpy as np 
from scipy.spatial.transform import Rotation as R
n_modes = 10
def dqs_Q(sQ, comp = True):
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
    return ret, Q_range

def euler_to_quat(euler, Q_range = None):
    q = euler.reshape((-1, 3))
    quats = np.zeros((n_modes * 2, 4))
    for i in range(n_modes):
        if Q_range is not None:
            qr = Q_range[0, i]
        else: 
            qr = 1.0
        rotation = R.from_euler('xyz', q[i] * 10 / qr, degrees = False)
        quats[i] = rotation.as_quat()
    return quats

def euler_to_affine(euler, Q_range = None):
    q = euler.reshape((-1, 3))
    affines = np.zeros((n_modes, 12))
    for i in range(n_modes):
        if Q_range is not None:
            qr = Q_range[0, i]
        else: 
            qr = 1.0
        rotation = R.from_euler('xyz', q[i] * 10 / qr, degrees = False)
        rr = rotation.as_matrix().T
        rrt = np.vstack([rr, np.zeros((1, 3))])
        affines[i] = rrt.flatten()
    return affines.flatten()
    
