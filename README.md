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


make sure you have the following dlls in the binary folder

```
cublas64_12.dll
cublasLt64_12.dll
cudart64_12.dll
cudss64_0.dll
cusparse64_12.dll
libiomp5md.dll
mkl_core.2.dll
mkl_intel_thread.2.dll
nvJitLink_120_0.dll
python310.dll
```