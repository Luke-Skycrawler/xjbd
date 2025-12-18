#### 
(Assume all commands are called in the root directory)
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
