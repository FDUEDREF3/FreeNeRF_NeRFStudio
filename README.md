# FreeNeRF implement with nerfstudio
An unofficial implementation (only implement freq_mask) for  [FreeNeRF: Improving Few-shot Neural Rendering with Free Frequency Regularization](https://arxiv.org/abs/2303.07418) based [nerfstudio](https://github.com/nerfstudio-project/nerfstudio) .

# how to use 

## install nerfstudio

follow the guide in [nerfstudio](https://github.com/nerfstudio-project/nerfstudio) to install nerfstudio.


## register freenerf

```python
 pip install -e .  
 ```

## train 

```python 
ns-train freenerfacto --data DATADIR
```

