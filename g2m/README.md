#### 
(Assume all commands are called in the root directory, and tensorboard is enabled (conda oldwarp env))
Process: 

#### generate data 
```
python -m g2m.gen_data
```
inputs: `data/36_d_2000_pi.npy`, 36 dimensional control rotations

outputs: `data/p_36d_2000_pi.npy`, corresponding c, r fitted with SQEM. 

The data generator is set to viewer mode to surf through the deformation and generate (c, r) on the fly, when it finishes it saves the .npy file. 

#### view generated data
```
python -m g2m.gen_data
```
make sure to call examine_data()

#### train

```
tensorboard --logdir runs
python -m g2m.train
```

### verify training results

```
python -m g2m.verify_nn
```

if autoplay is enabled, it will interpolate through the validate dataset. When it finishes, it produces q trajectory and `c, r` output from nn, stored as `q_traj.npy` and `p_traj.npy`. The deformed tet mesh node positions is stored as `v_traj.npy`.  