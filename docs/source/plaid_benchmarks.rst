PLAID Benchmarks
================

.. image:: images/plaid_benchmarks.png
    :align: center
    :width: 600px
    :alt: PLAID Benchmarks


We provide interactive benchmarks hosted on Hugging Face, in which anyone can test their own SciML method.
These benchmarks involve regression problems posed on datasets provided in PLAID format.
Some of these datasets have been introduced in the MMGP (Mesh Morphing Gaussian Process) paper :cite:p:`casenave2023mmgp`,
and the PLAID paper :cite:p:`casenave2025plaid`.
A ranking is automatically updated based on a score computed on the testing set of each dataset.
For the benchmarks to be meaningful, the outputs on the testing sets are not made public.

The relative RMSE is the considered metric for comparing methods. Let :math:`\{ \mathbf{U}^i_{\rm ref} \}_{i=1}^{n_\star}`
and :math:`\{ \mathbf{U}^i_{\rm pred} \}_{i=1}^{n_\star}` be the test observations and predictions, respectively, of a given field of interest.
The relative RMSE is defined as

.. math::

    \mathrm{RRMSE}_f(\mathbf{U}_{\rm ref}, \mathbf{U}_{\rm pred}) = \left( \frac{1}{n_\star}\sum_{i=1}^{n_\star} \frac{\frac{1}{N^i}\|\mathbf{U}^i_{\rm ref} - \mathbf{U}^i_{\rm pred}\|_2^2}{\|\mathbf{U}^i_{\rm ref}\|_\infty^2} \right)^{1/2},

where :math:`N^i` is the number of nodes in the mesh :math:`i`, and :math:`\max(\mathbf{U}^i_{\rm ref})` is the maximum entry in the vector :math:`\mathbf{U}^i_{\rm ref}`. Similarly for scalar outputs:

.. math::

    \mathrm{RRMSE}_s(\mathbf{w}_{\rm ref}, \mathbf{w}_{\rm pred}) = \left( \frac{1}{n_\star} \sum_{i=1}^{n_\star} \frac{|w^i_{\rm ref} - w_{\rm pred}^i|^2}{|w^i_{\rm ref}|^2} \right)^{1/2}.


Resources
---------

+---------------------+-------------------------------------------+-----------------------------------------------+
|                     |           Dataset                         |                  Benchmark                    |
+---------------------+-------------------------------------------+-----------------------------------------------+
| **Tensile2d**       | |Tensile2d_HF| |Tensile2d_Z|              | |Tensile2d_Be|                                |
+---------------------+-------------------------------------------+-----------------------------------------------+
| **2D_MultiScHypEl** | |2D_MultiScHypEl_HF| |2D_MultiScHypEl_Z|  | |2D_MultiScHypEl_Be|                          |
+---------------------+-------------------------------------------+-----------------------------------------------+
| **2D_ElPlDynamics** | |2D_ElPlDynamics_HF| |2D_ElPlDynamics_Z|  | |2D_ElPlDynamics_Be|                          |
+---------------------+-------------------------------------------+-----------------------------------------------+
| **Rotor37**         | |Rotor37_HF|   |Rotor37_Z|                | |Rotor37_Be|                                  |
+---------------------+-------------------------------------------+-----------------------------------------------+
| **2D_profile**      | |2D_profile_HF| |2D_profile_Z|            | |2D_profile_Be|                               |
+---------------------+-------------------------------------------+-----------------------------------------------+
| **VKI-LS59**        | |VKI-LS59_HF| |VKI-LS59_Z|                | |VKI-LS59_Be|                                 |
+---------------------+-------------------------------------------+-----------------------------------------------+


.. |Tensile2d_Z| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.14840177.svg
  :target: https://doi.org/10.5281/zenodo.14840177

.. |Tensile2d_HF| image:: https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md-dark.svg
  :target: https://huggingface.co/datasets/PLAID-datasets/Tensile2d

.. |Tensile2d_Be| image:: https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg
  :target: https://huggingface.co/spaces/PLAIDcompetitions/Tensile2dBenchmark


.. |2D_MultiScHypEl_Z| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.14840446.svg
  :target: https://doi.org/10.5281/zenodo.14840446

.. |2D_MultiScHypEl_HF| image:: https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md-dark.svg
  :target: https://huggingface.co/datasets/PLAID-datasets/2D_Multiscale_Hyperelasticity

.. |2D_MultiScHypEl_Be| image:: https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg
  :target: https://huggingface.co/spaces/PLAIDcompetitions/2DMultiscaleHyperelasticityBenchmark


.. |2D_ElPlDynamics_Z| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.15286369.svg
  :target: https://doi.org/10.5281/zenodo.15286369

.. |2D_ElPlDynamics_HF| image:: https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md-dark.svg
  :target: https://huggingface.co/datasets/PLAID-datasets/2D_ElastoPlastoDynamics

.. |2D_ElPlDynamics_Be| image:: https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg
  :target: https://huggingface.co/spaces/PLAIDcompetitions/2DElastoPlastoDynamics


.. |Rotor37_Z| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.14840190.svg
  :target: https://doi.org/10.5281/zenodo.14840190

.. |Rotor37_HF| image:: https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md-dark.svg
  :target: https://huggingface.co/datasets/PLAID-datasets/Rotor37

.. |Rotor37_Be| image:: https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg
  :target: https://huggingface.co/spaces/PLAIDcompetitions/Rotor37Benchmark


.. |2D_profile_Z| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.15155119.svg
  :target: https://doi.org/10.5281/zenodo.15155119

.. |2D_profile_HF| image:: https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md-dark.svg
  :target: https://huggingface.co/datasets/PLAID-datasets/2D_profile

.. |2D_profile_Be| image:: https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg
  :target: https://huggingface.co/spaces/PLAIDcompetitions/2DprofileBenchmark


.. |VKI-LS59_Z| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.14840512.svg
  :target: https://doi.org/10.5281/zenodo.14840512

.. |VKI-LS59_HF| image:: https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md-dark.svg
  :target: https://huggingface.co/datasets/PLAID-datasets/VKI-LS59

.. |VKI-LS59_Be| image:: https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg
  :target: https://huggingface.co/spaces/PLAIDcompetitions/VKILS59Benchmark


AirfRANS, introduced in :cite:p:`airfrans` is an additional dataset provided in PLAID format and various variants.
Since the outputs on the testing sets are public, no benchmark application is provided for this dataset.

+-----------------------+--------------------------------+
| **AirfRANS original** | |AirfRANS_O_HF| |AirfRANS_O_Z| |
+-----------------------+--------------------------------+
| **AirfRANS clipped**  | |AirfRANS_C_HF| |AirfRANS_C_Z| |
+-----------------------+--------------------------------+
| **AirfRANS remeshed** | |AirfRANS_R_HF| |AirfRANS_R_Z| |
+-----------------------+--------------------------------+


.. |AirfRANS_O_Z| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.14840387.svg
  :target: https://doi.org/10.5281/zenodo.14840387

.. |AirfRANS_O_HF| image:: https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md-dark.svg
  :target: https://huggingface.co/datasets/PLAID-datasets/AirfRANS_original


.. |AirfRANS_C_Z| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.14840377.svg
  :target: https://doi.org/10.5281/zenodo.14840377

.. |AirfRANS_C_HF| image:: https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md-dark.svg
  :target: https://huggingface.co/datasets/PLAID-datasets/AirfRANS_clipped


.. |AirfRANS_R_Z| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.14840388.svg
  :target: https://doi.org/10.5281/zenodo.14840388

.. |AirfRANS_R_HF| image:: https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md-dark.svg
  :target: https://huggingface.co/datasets/PLAID-datasets/AirfRANS_remeshed




References
----------

.. bibliography::
    :style: unsrt
