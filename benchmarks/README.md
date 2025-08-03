## PLAID Benchmarks

This folder contains the code used to generate the baselines for the [PLAID Benchmarks](https://huggingface.co/PLAIDcompetitions). These benchmarks are interactive: anyone can participate and submit their own model predictions. Each benchmark application provides the corresponding dataset, along with documentation on how to use it and submit a solution.

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
The results on Aug, 3rd, 2025 are:
 | Dataset           | MGN | MMGP | Vi-Transf. | Augur | FNO | MARIO |
|-------------------|-----|------|------------|-------|-------|-------|
| `Tensile2d`       |  0.0673 |  **0.0026**  |   0.0058     |  0.0154   |  0.021  |  *0.0038*  |
| `2D_MultiScHypEl` | 0.0437  |  ❌  |   *0.0341*     |  **0.0232**   |   0.0439  |  0.0573  |
| `2D_ElPlDynamics` | 🕑  |  ❌  |   🕑     |  🕑    |  **0.0158**  |  🕑  |
| `Rotor37`         | 0.0074  |  **0.0014**  |   0.0029     |  0.0033   |   0.0313  |  *0.0017*  |
| `2D_profile`      |  0.0593 |  0.0365  |   *0.0319*     |  0.0425   |  0.0972  |  **0.0307**  |
| `VKI-LS59`        | 0.0684  |  0.0312  |   0.0493     |  *0.0267*   |   0.0581  |  **0.0124**  |

Additional notes:
- MARIO is computationally expensive to train, but delivers excellent accuracy across all datasets, demonstrating its high versatility.
- FNO struggles on datasets with strong mesh anisotropies (e.g., Rotor37 and 2D_profile): the need to project to and from regular grids leads to a significant loss of accuracy. On Rotor37, using a 3D regular grid also incurs high computational costs.