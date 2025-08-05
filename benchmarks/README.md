## PLAID Benchmarks

This folder contains the code used to generate the baselines for the [PLAID Benchmarks](https://huggingface.co/PLAIDcompetitions), with the exception of the Augur results, which are produced using a [commercial solution](https://augurco.fr/). These benchmarks are interactive: anyone can participate by submitting their own model predictions. Each benchmark includes the corresponding dataset, along with documentation on how to use it and how to submit a solution.


### Benchmark results, as of August 5, 2025:
| Dataset           | MGN | MMGP | Vi-Transf. | Augur | FNO | MARIO |
|-------------------|-----|------|------------|-------|-------|-------|
| `Tensile2d`       | 0.0673  |  **0.0026**  |   0.0116     |  0.0154   |  0.0123  |  *0.0038*  |
| `2D_MultiScHypEl` | 0.0437  |  ❌  |   0.0325     |  **0.0232**   |   *0.0302*  |  0.0573  |
| `2D_ElPlDynamics` | 0.1202  |  ❌  |   *0.0227*     |  0.0346    |  **0.0215**  |  0.0319  |
| `Rotor37`         | 0.0074  |  **0.0014**  |   0.0029     |  0.0033   |   0.0313  |  *0.0017*  |
| `2D_profile`      | 0.0593  |  0.0365  |   *0.0312*     |  0.0425   |  0.0972  |  **0.0307**  |
| `VKI-LS59`        | 0.0684  |  0.0312  |   *0.0193*     |  0.0267    |   0.0215  |  **0.0124**  |

❌: Not compatible with topology variation

### Additional notes:
- **MMGP** does not support variable mesh topologies, which limits its applicability to certain datasets and often necessitates custom preprocessing for new cases. However, when morphing is either unnecessary or inexpensive, it offers a highly efficient solution, combining fast training with good accuracy (e.g., `Tensile2d` and `Rotor37`).
- **MARIO** is computationally expensive to train but achieves consistently a very strong performance across most datasets. Its results on `2D_MultiScHypEl` and `2D_ElPlDynamics` are slightly worse than other tested methods, which may reflect the challenge of capturing complex shape variability in these cases.
- **Vi-Transformer** and **Augur** perform well across all datasets, showing strong versatility and generalization capabilities.
- **FNO** suffers on datasets featuring unstructured meshes with pronounced anisotropies, due to the loss of accuracy introduced by projections to and from regular grids (e.g., `Rotor37` and `2D_profile`). Additionally, the use of a 3D regular grid on `Rotor37` results in substantial computational overhead.
