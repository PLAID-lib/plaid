# PLAID Benchmarks

![PLAID Benchmarks](images/plaid_benchmarks.png){ width="600" }

We provide interactive benchmarks hosted on Hugging Face, in which anyone can test their own SciML method.
These benchmarks involve regression problems posed on datasets provided in PLAID format.
Some of these datasets have been introduced in the MMGP (Mesh Morphing Gaussian Process) paper and the PLAID paper.
A ranking is automatically updated based on a score computed on the testing set of each dataset.
For the benchmarks to be meaningful, the outputs on the testing sets are not made public.

The relative RMSE is the considered metric for comparing methods. Let $\{ \mathbf{U}^i_{\rm ref} \}_{i=1}^{n_\star}$
and $\{ \mathbf{U}^i_{\rm pred} \}_{i=1}^{n_\star}$ be the test observations and predictions, respectively, of a given field of interest.
The relative RMSE is defined as

$$
\mathrm{RRMSE}_f(\mathbf{U}_{\rm ref}, \mathbf{U}_{\rm pred}) = \left( \frac{1}{n_\star}\sum_{i=1}^{n_\star} \frac{\frac{1}{N^i}\|\mathbf{U}^i_{\rm ref} - \mathbf{U}^i_{\rm pred}\|_2^2}{\|\mathbf{U}^i_{\rm ref}\|_\infty^2} \right)^{1/2},
$$

where $N^i$ is the number of nodes in the mesh $i$, and $\max(\mathbf{U}^i_{\rm ref})$ is the maximum entry in the vector $\mathbf{U}^i_{\rm ref}$. Similarly for scalar outputs:

$$
\mathrm{RRMSE}_s(\mathbf{w}_{\rm ref}, \mathbf{w}_{\rm pred}) = \left( \frac{1}{n_\star} \sum_{i=1}^{n_\star} \frac{|w^i_{\rm ref} - w_{\rm pred}^i|^2}{|w^i_{\rm ref}|^2} \right)^{1/2}.
$$

## Resources

|                     | Dataset | Benchmark |
|---------------------|---------|-----------|
| **Tensile2d**       | [![HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md-dark.svg)](https://huggingface.co/datasets/PLAID-datasets/Tensile2d) [![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.14840177.svg)](https://doi.org/10.5281/zenodo.14840177) | [![Space](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg)](https://huggingface.co/spaces/PLAIDcompetitions/Tensile2dBenchmark) |
| **2D_MultiScHypEl** | [![HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md-dark.svg)](https://huggingface.co/datasets/PLAID-datasets/2D_Multiscale_Hyperelasticity) [![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.14840446.svg)](https://doi.org/10.5281/zenodo.14840446) | [![Space](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg)](https://huggingface.co/spaces/PLAIDcompetitions/2DMultiscaleHyperelasticityBenchmark) |
| **2D_ElPlDynamics** | [![HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md-dark.svg)](https://huggingface.co/datasets/PLAID-datasets/2D_ElastoPlastoDynamics) [![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.15286369.svg)](https://doi.org/10.5281/zenodo.15286369) | [![Space](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg)](https://huggingface.co/spaces/PLAIDcompetitions/2DElastoPlastoDynamics) |
| **Rotor37**         | [![HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md-dark.svg)](https://huggingface.co/datasets/PLAID-datasets/Rotor37) [![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.14840190.svg)](https://doi.org/10.5281/zenodo.14840190) | [![Space](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg)](https://huggingface.co/spaces/PLAIDcompetitions/Rotor37Benchmark) |
| **2D_profile**      | [![HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md-dark.svg)](https://huggingface.co/datasets/PLAID-datasets/2D_profile) [![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.15155119.svg)](https://doi.org/10.5281/zenodo.15155119) | [![Space](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg)](https://huggingface.co/spaces/PLAIDcompetitions/2DprofileBenchmark) |
| **VKI-LS59**        | [![HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md-dark.svg)](https://huggingface.co/datasets/PLAID-datasets/VKI-LS59) [![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.14840512.svg)](https://doi.org/10.5281/zenodo.14840512) | [![Space](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg)](https://huggingface.co/spaces/PLAIDcompetitions/VKILS59Benchmark) |

AirfRANS is an additional dataset provided in PLAID format and various variants.

| Dataset | Links |
|---------|-------|
| **AirfRANS original** | [![HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md-dark.svg)](https://huggingface.co/datasets/PLAID-datasets/AirfRANS_original) [![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.14840387.svg)](https://doi.org/10.5281/zenodo.14840387) |
| **AirfRANS clipped**  | [![HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md-dark.svg)](https://huggingface.co/datasets/PLAID-datasets/AirfRANS_clipped) [![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.14840377.svg)](https://doi.org/10.5281/zenodo.14840377) |
| **AirfRANS remeshed** | [![HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md-dark.svg)](https://huggingface.co/datasets/PLAID-datasets/AirfRANS_remeshed) [![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.14840388.svg)](https://doi.org/10.5281/zenodo.14840388) |

## Benchmark results

As of August 5, 2025.

| Dataset | MGN | MMGP | Vi-Transf. | Augur | FNO | MARIO |
|---|---:|---:|---:|---:|---:|---:|
| `Tensile2d` | 0.0673 | 0.0026 | 0.0116 | 0.0154 | 0.0123 | 0.0038 |
| `2D_MultiScHypEl` | 0.0437 | ❌ | 0.0325 | 0.0232 | 0.0302 | 0.0573 |
| `2D_ElPlDynamics` | 0.1202 | ❌ | 0.0227 | 0.0346 | 0.0215 | 0.0319 |
| `Rotor37` | 0.0074 | 0.0014 | 0.0029 | 0.0033 | 0.0313 | 0.0017 |
| `2D_profile` | 0.0593 | 0.0365 | 0.0312 | 0.0425 | 0.0972 | 0.0307 |
| `VKI-LS59` | 0.0684 | 0.0312 | 0.0193 | 0.0267 | 0.0215 | 0.0124 |

❌: Not compatible with topology variation.

!!! note
    - MMGP does not support variable mesh topologies, which limits its applicability to certain datasets and often necessitates custom preprocessing for new cases. However, when morphing is either unnecessary or inexpensive, it offers a highly efficient solution, combining fast training with good accuracy (e.g., `Tensile2d` and `Rotor37`).
    - MARIO is computationally expensive to train but achieves consistently a very strong performance across most datasets. Its result on `2D_MultiScHypEl` is slightly worse than other tested methods, which may reflect the challenge of capturing complex shape variability in these cases.
    - Vi-Transformer and Augur perform well across all datasets, showing strong versatility and generalization capabilities.
    - FNO suffers on datasets featuring unstructured meshes with pronounced anisotropies, due to the loss of accuracy introduced by projections to and from regular grids (e.g., `Rotor37` and `2D_profile`). Additionally, the use of a 3D regular grid on `Rotor37` results in substantial computational overhead.
