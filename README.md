### Warp Simulation Library


#### XPBD 

command: 
```
python xpbd.py
```


#### Sifakis FEM 

1. to visualize eigen modes:
command:

```
python vis_eigs.py
```
1. to drape an elastic bar with line search newton solver:
```
python stretch.py
```
material model subject to specification by
`python
from .neo_hookean import PK1, tangent_stiffness, psi
` 

#### Modal Warping

```
python modal_warping.py
```
