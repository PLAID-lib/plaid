## PLAID Benchmarks

This folder contains the code used to generate the baselines for the [PLAID Benchmarks](https://huggingface.co/PLAIDcompetitions), with the exception of the Augur results, which are produced using a [commercial solution](https://augurco.fr/). These benchmarks are interactive: anyone can participate by submitting their own model predictions. Each benchmark includes the corresponding dataset, along with documentation on how to use it and how to submit a solution.

<!-- | Dataset           | MGN | MMGP | Vi-Transf. | Augur | FNO | MARIO |
|-------------------|-----|------|------------|-------|-------|-------|
| `Tensile2d`       | 🔵  |  🔵  |   🔵     |  🔵   |   ✅  |  ✅  |
| `2D_MultiScHypEl` | 🔵  |  ❌  |   🔵     |  🔵   |   ✅  |  ✅  |
| `2D_ElPlDynamics` | 🕑  |  ❌  |   🕑     |  🕑    |  🔵  |  🕑  |
| `Rotor37`         | 🔵  |  🔵  |   🔵     |  🔵   |   ✅  |  ✅  |
| `2D_profile`      | 🔵  |  🔵  |   🔵     |  ✅   |   ✅  |  ✅  |
| `VKI-LS59`        | 🔵  |  🔵  |   🔵     |  🔵   |   ✅  |  🔵  |
- 🔵: Present in initial submission
- ✅: Added post-submission on Hugging Face
- ❌: Not compatible with topology variation
  -->

**Results as of August 3, 2025:**
 | Dataset           | MGN | MMGP | Vi-Transf. | Augur | FNO | MARIO |
|-------------------|-----|------|------------|-------|-------|-------|
| `Tensile2d`       |  0.0673 |  **0.0026**  |   0.0058     |  0.0154   |  0.021  |  *0.0038*  |
| `2D_MultiScHypEl` | 0.0437  |  ❌  |   *0.0341*     |  **0.0232**   |   0.0439  |  0.0573  |
| `2D_ElPlDynamics` | 🕑  |  ❌  |   🕑     |  🕑    |  **0.0158**  |  🕑  |
| `Rotor37`         | 0.0074  |  **0.0014**  |   0.0029     |  0.0033   |   0.0313  |  *0.0017*  |
| `2D_profile`      |  0.0593 |  0.0365  |   *0.0319*     |  0.0425   |  0.0972  |  **0.0307**  |
| `VKI-LS59`        | 0.0684  |  0.0312  |   0.0493     |  *0.0267*   |   0.0581  |  **0.0124**  |

**Additional notes:**
- **MMGP** does not support variable mesh topologies, which prevents its application to some datasets. However, when morphing is either unnecessary or inexpensive, it offers a highly efficient solution, combining fast training with good accuracy (e.g., `Tensile2d`, `Rotor37`).
- **MARIO** is computationally expensive to train but achieves consistently strong performance across most datasets. Its result on `2D_MultiScHypEl` is slightly higher than other top performers, which may reflect the challenge of capturing complex shape variability in that case.
- **Vi-Transformer** and **Augur** perform well across all datasets, showing strong versatility and generalization capabilities.
- **FNO** suffers significantly on datasets with strong mesh anisotropies: the projections to and from regular grids degrade accuracy, especially on datasets with strong mesh anisotropies such as `Rotor37` and `2D_profile`. Additionally, the use of a 3D regular grid on `Rotor37` results in substantial computational overhead.
