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