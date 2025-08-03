## PLAID Benchmarks

This folder contains the code used to generate the baselines for the [PLAID Benchmarks](https://huggingface.co/PLAIDcompetitions), with the exception of the Augur results, which are produced using a [commercial solution](https://augurco.fr/). These benchmarks are interactive: anyone can participate by submitting their own model predictions. Each benchmark includes the corresponding dataset, along with documentation on how to use it and how to submit a solution.

**As of August 3, 2025:**

### Code distribution status:
| Dataset           | MGN | MMGP | Vi-Transf. | FNO | MARIO |
|-------------------|-----|------|------------|-------|-------|
| `Tensile2d`       | 🕑  |  ✅  |   🕑     |  ✅  |  🕑  |
| `2D_MultiScHypEl` | 🕑  |  ❌  |   🕑     |  ✅  |  🕑  |
| `2D_ElPlDynamics` | 🕑  |  ❌  |   🕑     |  🕑  |  🕑  |
| `Rotor37`         | 🕑  |  ✅  |   🕑     |  ✅  |  ✅  |
| `2D_profile`      | 🕑  |  🕑  |   🕑     |  ✅  |  ✅  |
| `VKI-LS59`        | 🕑  |  ✅  |   🕑     |  ✅  |  ✅  |
- ❌: Not compatible with topology variation

<!-- We thank the reviewer for their encouraging feedback and for responding early in the discussion period. This gave us both the opportunity and the motivation to complete the remaining work in time to provide a full response before the end of the discussion phase. As a result, we were able to finalize the benchmark table and release the code online.

We believe this additional work fully addresses points W1 and W2 raised in the initial review. The updated table is shown below (displaying only the `total_error`), and each entry corresponds to a submission that can be consulted on the Hugging Face interactive benchmarks. The code to reproduce these results (excluding **Augur**, which relies on a commercial solution and cannot be open-sourced) is available in the `benchmarks` folder of the PLAID repository. -->

### Benchmark status:
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
- **MARIO** is computationally expensive to train but achieves consistently a very strong performance across most datasets. Its result on `2D_MultiScHypEl` is slightly worse than other tested methods, which may reflect the challenge of capturing complex shape variability in that case.
- **Vi-Transformer** and **Augur** perform well across all datasets, showing strong versatility and generalization capabilities.
- **FNO** suffers significantly on datasets with strong mesh anisotropies: the projections to and from regular grids degrade accuracy, especially on datasets with strong mesh anisotropies such as `Rotor37` and `2D_profile`. Additionally, the use of a 3D regular grid on `Rotor37` results in substantial computational overhead.


<!-- Longuer version (not a good idea, imho)

We would like to apologize for the confusing presentation and lack of clear distinction in our rebuttal between “work in progress” and the intended outlook for this project. By “work in progress,” we meant that the benchmark table would be completed for the camera-ready version, and that all code necessary to reproduce the baseline results would be made publicly available in the PLAID repository — with the exception of Augur, which relies on a commercial solution and cannot be open-sourced. Most of the relevant code was already implemented at the time of our rebuttal. The two additional time-dependent datasets — crack propagation in structural mechanics and unsteady turbulent CFD simulations — were meant as future extensions of our work. They were not mentioned in the initial submission and were intended as part of the broader outlook rather than ongoing benchmarks.

We thank the reviewer for their encouraging feedback and for responding early in the discussion period. This gave us both the opportunity and the motivation to complete the remaining work in time to provide a full response before the end of the discussion phase. As a result, we were able to finalize the benchmark table and release the code online.

We believe this additional work fully addresses points W1 and W2 raised in the initial review. The updated table is shown below (displaying only the `total_error`), and each entry corresponds to a submission that can be consulted on the Hugging Face interactive benchmarks. The code to reproduce these results (excluding **Augur**) is available in the `benchmarks` folder of the PLAID repository. -->



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
