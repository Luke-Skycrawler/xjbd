# biharmonic weights compute
import numpy as np  
import warp as wp 
import igl
import matplotlib.pyplot as plt
import matplotlib.tri as tri

V, _, N, F, _, _ = igl.read_obj("assets/alligator.obj")
C, BE,_, _, _, _ = igl.read_tgf("assets/alligator-skeleton-cage-points.tgf")

print(C.shape, BE.shape)
z0 = np.zeros(0)
z00 = np.zeros((0, 0))
_, b, bc = igl.boundary_conditions(V, F, C, z0, BE, z00)
print(b.shape, bc.shape)
print(b, "\n\n", bc)
triangles = tri.Triangulation(V[:, 0], V[:, 1], F)

fig, ax = plt.subplots()
ax.triplot(triangles, 'r-', alpha = 0.6)
ax.scatter(V[:, 0], V[:, 1], c = 'black', s = 5)


bbx = np.max(V[:, 0]) - np.min(V[:, 1])
bby = np.max(V[:, 1]) - np.min(V[:, 1])
bb = max(bbx, bby)

selected = []
display_verts = None
def on_click(event):
    global selected, display_verts
    if event.inaxes:
        x, y = event.xdata, event.ydata
        distances = np.linalg.norm(V[:, :2] - np.array([x, y]), axis = 1)
        idx = np.argmin(distances)
        if distances[idx] < 1e-2 * bb:
            selected = []
            selected.append(idx)
            if display_verts is not None:
                display_verts.remove()
            display_verts = ax.scatter(V[selected, 0], V[selected, 1], c = 'blue', s = 20)
            fig.canvas.draw()

fig.canvas.mpl_connect('button_press_event', on_click)
plt.axis('equal')
plt.show()