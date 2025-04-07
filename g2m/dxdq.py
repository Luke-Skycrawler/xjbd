import numpy as np
def dxdq_jacobian(n_nodesx3, V):
    n_nodes = n_nodesx3 // 3
    q6 = np.zeros((n_nodesx3, 6))
    skew = lambda x: np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
    print(f"n_nodes: {n_nodes}")
    for i in range(n_nodes):
        q6[3 * i: 3 * i + 3, : 3] = np.eye(3)
    for i in range(V.shape[0]):
        q6[3 * i: 3 * i + 3, 3:] = skew(V[i])
    return q6
