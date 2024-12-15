import numpy as np
import os
def export_tobj(file, V, T):
    to_strv = lambda x: f"v {x[0]} {x[1]} {x[2]}\n"
    to_strt = lambda t: f"t {t[0]} {t[1]} {t[2]} {t[3]}\n"

    linesv = [to_strv(v) for v in V] 
    linest = [to_strt(t) for t in T]
    with open(file, 'w') as f:
        f.writelines(linesv + linest)


def import_tobj(filename): 
    vertices = []
    faces = []
    if not os.path.exists(filename):
        print(f"{filename} does not exist")
        return None, None
    with open(filename, 'r') as file:
        for line in file:
            # Split the line into components
            parts = line.split()

            if not parts:
                continue

            # Parse vertex (v x y z)
            if parts[0] == 'v':
                vertex = list(map(float, parts[1:4]))  # Extract x, y, z coordinates
                vertices.append(np.array(vertex))

            # Parse face (f v1 v2 v3)
            elif parts[0] == 't':
                # Convert to zero-based indexing by subtracting 1 from each index
                face = [int(idx) for idx in parts[1:]]
                faces.append(np.array(face))

    V = np.array(vertices)
    T = np.array(faces)
    print(V.shape, T.shape)
    return V, T