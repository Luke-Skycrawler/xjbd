import numpy as np
import os
import shutil
if __name__ == "__main__":
    # stratified sampling
    end_frame = 2000
    n_samples = 5
    samples = np.zeros((n_samples, ), dtype=int)
    for i in range(n_samples):  
        np.random.seed(i * 100 + 42)
        start = i * end_frame // n_samples
        end = (i + 1) * end_frame // n_samples
        rs = np.random.randint(start, end)
        samples[i] = rs

    path = "output/waterwheel/61_random_20modes/states"
    dst_path = "output/states"
    for s in samples:
        source = os.path.join(path, f"z_{s}.npz")
        dst = os.path.join(dst_path, f"z_{s}.npz")
        shutil.copyfile(source, dst)
    

    