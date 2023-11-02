Data challenges
===============

As part of the documentation for PLAID, we detail the three numerical experiments of the paper:
"MMGP: Mesh Morphing Gaussian Process-based machine learning method
for regression of physical problems under non-parameterized geometrical variability"
(`arXiv preprint <https://arxiv.org/abs/2305.12871>`_, `neurips 2023 <https://arxiv.org/abs/2305.12871>`_).
The library implementating the MMGP method is available at `Safran Gitlab <https://gitlab.com/drti/mmgp>`_,
and documented on `readthedocs <https://mmgp.readthedocs.io/>`_.

The corresponding datasets are available in PLAID format on `Zenodo <https://zenodo.org/>`_.


Following the notations of the paper, let :math:`\{ \mathbf{U}^i_{\rm ref} \}_{i=1}^{n_\star}` and :math:`\{ \mathbf{U}^i_{\rm pred} \}_{i=1}^{n_\star}` be respectively test observations and predictions of a given field of interest.
The relative RMSE is defined as

.. math::

    \mathrm{RRMSE}_f(\mathbf{U}_{\rm ref}, \mathbf{U}_{\rm pred}) = \left( \frac{1}{n_\star}\sum_{i=1}^{n_\star} \frac{\frac{1}{N^i}\|\mathbf{U}^i_{\rm ref} - \mathbf{U}^i_{\rm pred}\|_2^2}{\|\mathbf{U}^i_{\rm ref}\|_\infty^2} \right)^{1/2},

where :math:`N^i` is the number of nodes in the mesh :math:`i`, and :math:`\max(\mathbf{U}^i_{\rm ref})` is the maximum entry in the vector :math:`\mathbf{U}^i_{\rm ref}`. Similarly for scalar outputs:

.. math::

    \mathrm{RRMSE}_s(\mathbf{w}_{\rm ref}, \mathbf{w}_{\rm pred}) = \left( \frac{1}{n_\star} \sum_{i=1}^{n_\star} \frac{|w^i_{\rm ref} - w_{\rm pred}^i|^2}{|w^i_{\rm ref}|^2} \right)^{1/2}.

For the challenges, we define a composite score :math:`S` as the mean of all the field and scalar RRMSE, computed on the test split:

.. math::

    S = \frac{1}{n_f+n_s}\left(\sum_{f \in {\rm fields}} \mathrm{RRMSE}_f(\mathbf{U}_{\rm ref}, \mathbf{U}_{\rm pred})+
    \sum_{s \in {\rm scalars}} \mathrm{RRMSE}_s(\mathbf{w}_{\rm ref}, \mathbf{w}_{\rm pred})\right),

where :math:`n_f` and :math:`n_s` are respectively the number of ouput fields and scalars of interest.

Data challenges:

.. toctree::
    :glob:
    :maxdepth: 1

    data_challenges/tensile2d
    data_challenges/rotor37
    data_challenges/airfrans
