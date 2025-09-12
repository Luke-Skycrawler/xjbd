from utils.tobj import import_tobj, export_tobj
from g2m.medial import SlabMesh


def scale_tobj(file, scale):
    input = f"assets/{file}/{file}.tobj"
    output = f"assets/{file}/{file}_scaled.tobj"
    V, T = import_tobj(input)
    export_tobj(output, V * scale, T)
    
def scale_ma(file, scale):
    input = f"assets/{file}/ma/{file}.ma"
    output = f"assets/{file}/ma/{file}_scaled.ma"
    mesh = SlabMesh(input)
    mesh.V *= scale
    mesh.R *= scale
    mesh.export_ma(output)  

def scale_all(file, scale):
    scale_tobj(file, scale)
    scale_ma(file, scale)


if __name__ == "__main__":
    scale_all("boatv9", 0.1)
    