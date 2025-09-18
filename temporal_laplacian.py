import numpy as np 
import os
import matplotlib.pyplot as plt
import scipy.sparse as sp
n_frames = 800
hip = "D:/tencent work space/repos/tet converter"

def load(scene = "C2_v9.3"):
    z = np.zeros((n_frames, 12), float)
    for ff in range(1, n_frames):
        frame = ff * 4
        file = f"states/{scene}/states/z_{frame}.npz"
        path = os.path.join(hip, file)
        zt = np.load(path)["z"][:12]
        z[ff] = zt
    return z
def smooth(z, n_smooth_iters = 10):
    for i in range(n_smooth_iters):
        for ff in range(1, n_frames):
            if ff %  2 == 0:
                z_last = z[max(1, ff - 1)]
                z_nxt = z[min(n_frames - 1, ff + 1)]
                z[ff] = 0.5 * (z_last + z_nxt)
        for ff in range(1, n_frames):
            if ff %  2 == 1:
                z_last = z[max(1, ff - 1)]
                z_nxt = z[min(n_frames - 1, ff + 1)]
                z[ff] = 0.5 * (z_last + z_nxt)
    return z

if __name__ == "__main__":
    z = load()
    zt = np.copy(z)
    zt = smooth(zt, n_smooth_iters = 500)
    plt.plot(z[:, 11], label = "original")
    plt.plot(zt[:, 11], label = "smoothed")
    plt.legend()
    plt.show()
    np.save("data/z_smooth.npy", zt)

    # geo.addAttrib(hou.attribType.Global, "z", z.reshape(-1))
    # print(zt)