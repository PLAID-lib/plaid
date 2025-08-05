# PLAID Vision Transformer benchmarks

## Training and submission
The Vi-Transformer benchmark models can be ran on the corresponding datasets as follows:

### Stationary datasets

#### 2D_MultiScHypEl
```
python main_static.py --config-name 2d_multiscale_hyperelasticity.yaml
```

#### 2D_profile
```
python main_static.py --config-name 2d_profile.yaml
```

#### Rotor37
```
python main_static.py --config-name rotor37.yaml
```

#### Tensile2d
```
python main_static.py --config-name tensile2d.yaml
```

#### VKI-LS59
```
python main_static.py --config-name vkils59.yaml
```

### Non-stationary datasets

#### 2D_ElPlDynamics
```
python main_elasto_plasto_dynamics.py
```


## Dependencies
- [PLAID=0.1](https://github.com/PLAID-lib/plaid)
- [PyTorch=2.7.0](https://pytorch.org/)
- [Datasets=3.6.0](https://pypi.org/project/datasets/)
- [Einops=0.8.1](https://pypi.org/project/einops/)
- [Muscat=2.4.1](https://gitlab.com/drti/muscat)
- [PyTorchGeometric=2.6.1](https://pytorch-geometric.readthedocs.io/en/latest/)
- [Hydra=1.3.2](https://hydra.cc/docs/intro/)
- [Pymetis=2025.0.1](https://github.com/inducer/pymetis)
- [Omegaconf=2.3.0](https://omegaconf.readthedocs.io/en/2.3_branch/)
- [TorchTbProfiler=0.4.3](https://pypi.org/project/torch-tb-profiler/)
