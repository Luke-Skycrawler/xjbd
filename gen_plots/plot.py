import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_path = "../assets/fonts/linux_libertine/LinLibertine_RB.ttf"
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = prop.get_name()
plt.style.use("ggplot")
x = np.array([10, 20, 40, 64, 640])
y = np.array([3.47e-5, 1.81e-5, 1.38e-5, 1.21e-5, 5e-6])

t = []
terr = []
for a in x: 
    ti = np.load(f"../data/checkpoints/b{a}/timing_dict.npy")[1:]
    t.append(np.mean(ti))
    terr.append(np.std(ti))
    # print(ti)
fig, ax = plt.subplots(1, 1)
ax.plot(x, y, 'x-', color = 'C0')
ax.set_xlim(0, 70)
ax.set_xlabel("Hidden Layer Width")
ax.set_ylabel("Validation Loss")

# color2 = next(ax._get_lines.prop_cycler)['color']
t = np.array(t)
terr = np.array(terr)
print(t, terr)
ax2 = ax.twinx()
ax2.plot(x, t, '-', color = 'C1')
ax2.set_ylabel("Inference Time (ms)")
ax2.set_ylim(0, 2.5)
ax.set_ylim(1.0e-5,3.5e-5)
ax2.errorbar(x, t, yerr=terr, color = 'C1', capsize=5)
# ax2.tick_params(axis='y', labelcolor='C1')
# from brokenaxes import brokenaxes 
# bax = brokenaxes(
#     xlims=((10, 70), (600, 680)),
#     hspace=0.05
# )

# bax.plot(x, y, marker='o')
plt.show()
