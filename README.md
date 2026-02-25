### Warp Simulation Library


#### XPBD 

command: 
```
python xpbd.py
```


#### Sifakis FEM 

command:

```
python vis_eigs.py
```

#### Modal Warping

```
python modal_warping.py
```

#### Collision Test 
```
python -m geometry.collision_cell
```


#### Raining Bunnies

FEM simulation of 20 bunnies, with neo-hookean elasticity (FIXME: no psd projection) & analytically projected [Shi 2023] penalty-based collision.

Screenshot output folder default to `output/`

```
python mesh_complex.py
```
#### Fast Complimentary Dynamics

Implements unconstrained "Fast Complementary Dynamics via Skinning Eigenmodes" weights generation. 
```
python fast_cd.py
```

#### Projective Dynamics

Implements basic PD simulator described in Quasi-Newton Methods for Real-time Simulation of Hyperelastic Materials. command: 
```
python admm-pd.py
```

#### Differentiable FEM 

Implement the rest shape optimization problem in Fig. 2 of SGN: Sparse Gauss-Newton for Accelerated Sensitivity Analysis. Command: 

```
python difffem.py
```
