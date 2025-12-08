import numpy as np
import matplotlib.pyplot as plt
class ConvergenceRecord: 
    def  __init__(self):
        self.records = []
    
    def append(self, frame, value):
        while len(self.records) <= frame:
            self.records.append([])
        self.records[frame].append(value)
    
    def save(self):
        # np.savez("convergence.npz", *self.records)
        np.save("convergence.npy", np.array(self.records))

    def plot(self, path = ""):
        plt.switch_backend("WebAgg")
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

        bins = [data[s1 == i] for i in range(np.max(s1))]


        n_iter = 8
        y_est = np.zeros((n_iter,))
        
        bin = np.array([np.array(i) for i in bins[n_iter]])
        bin /= bin[:, 0 : 1]
        bin = np.log10(bin)
        y_est = np.mean(bin, axis = 0)
        y_err = np.std(bin, axis = 0)
        # for sample in bins[n_iter]:
        #     sample /= sample[0]

        x = np.arange(n_iter)
        fig, ax = plt.subplots()

        ax.plot(x, y_est, '-')
        ax.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2)
        # ax.set_yscale('log')
        plt.show()
        # ax.plot(x, y, 'o', color='tab:brown')
        

if __name__ == "__main__":
    record = ConvergenceRecord()
    record.plot("plot/sbfgs")