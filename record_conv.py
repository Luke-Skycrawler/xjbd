import numpy as np
import matplotlib.pyplot as plt
class ConvergenceRecord: 
    def  __init__(self):
        self.records = []

        self.y_samples = 100
        self.m1 = np.zeros((self.y_samples))
        self.m2 = np.zeros((self.y_samples))
        self.cnt = 0
    
    def append(self, frame, value):
        while len(self.records) <= frame:
            self.records.append([])
        self.records[frame].append(value)
    
    def save(self):
        # np.savez("convergence.npz", *self.records)
        np.save("plot/convergence.npy", np.array(self.records, dtype = object))


    def convert_yline(self, xs, y_est, n_samples, ax):
        
        ymax = 0
        ymin = -5
        
        n_iters = len(xs)
        xs = np.concatenate([xs, [n_iters]])
        y_est = np.concatenate([y_est, [ymin]])

        ys = np.linspace(ymax, ymin, self.y_samples)
        x_inv = np.interp(ys, y_est[::-1], xs[::-1])
        
        self.m1 += x_inv * n_samples
        self.m2 += x_inv * x_inv * n_samples
        self.cnt += n_samples
        # ax.plot(x_inv, ys, '-')

    def plot(self, path = "", reuse_fig = False, show = True):
        
        self.m1[:] = 0.0
        self.m2[:] = 0.0
        self.cnt = 0

        # plt.switch_backend("WebAgg")
        if path != "":
            data = np.load(f"{path}/convergence.npy", allow_pickle=True)
        else:
            data = np.load("convergence.npy", allow_pickle=True)
        dim = len(data)
        s1 = np.zeros(dim, int)
        
        for i in range(1, dim):
            s1[i] = len(data[i])
        
        print(f"mean convergence iters = {np.mean(s1[1:])}")
        # bins = [[] for _ in range(20)]
        
        # for d in data: 
        #     for i, dd in enumerate(d):
        #         dd /= d[0]
        #         bins[i].append(np.log10(dd))
        
        # y_est = np.zeros((20, ))
        # y_err = np.zeros((20, ))

        # for i in range(1, 20):
        #     y_est[i] = np.mean(np.array(bins[i]))
        #     y_err[i] = np.std(np.array(bins[i]))

        bins = [data[s1 == i] for i in range(20)]

        if not reuse_fig: 
            self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)
        
        ax1 = self.ax1
        ax2 = self.ax2
        for n_iter in range(1, 20):
            # n_iter = 8
            y_est = np.zeros((n_iter,))
            
            if len(bins[n_iter]) == 0:
                continue
            bin = np.array([np.array(i) for i in bins[n_iter]])
            
            bin /= bin[:, 0 : 1]
            bin = np.log10(bin)
            y_est = np.mean(bin, axis = 0)
            y_err = np.std(bin, axis = 0)
            # for sample in bins[n_iter]:
            #     sample /= sample[0]

            x = np.arange(n_iter)

            ax1.plot(x, y_est, '-')
            ax1.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2)
            self.convert_yline(x, y_est, bins[n_iter].shape[0], ax2)
            # ax.set_yscale('log')

        # summary plot
        # fig, ax = plt.subplots()
        x_avg = self.m1 / self.cnt
        x_std = np.sqrt(self.m2 / self.cnt - x_avg * x_avg)
        y = np.linspace(0, -5, self.y_samples)
        ax2.plot(x_avg, y, '-', label = path.split("/")[-1] if path != "" else "data")
        ax2.fill_betweenx(y, x_avg - x_std, x_avg + x_std, alpha=0.2)
        
        
        if show:
            plt.legend()
            plt.show()
        # ax.plot(x, y, 'o', color='tab:brown')
    
    def plot_time(self, path):
        meta = np.load(f"{path}/metadata.npy", allow_pickle=True).item()
        print(meta)
        tot_iters = meta["total_iters"]
        tot_frames = meta["total_frames"]
        
        print(f"total iters = {tot_iters}, total frames = {tot_frames}")
        
        time = np.load(f"{path}/timeit.npz", allow_pickle=True)
        print(time["compute A"].shape)

if __name__ == "__main__":
    record = ConvergenceRecord()
    folders = ["plot/mbfgs_m2", "plot/mbfgs_m4", "plot/bfgs_m8", "plot/mbfgs_m8", "plot/mbfgs_m16", "plot/newton", "plot/frozen"]
    for i, folder in enumerate(folders):
        show = (i == len(folders) - 1)
        record.plot(folder, show = show, reuse_fig = i!= 0)
