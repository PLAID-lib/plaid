Tensile2d
=========


.. tabs::

    .. tab:: Dataset

        The dataset ``Tensile2d`` can be downloaded `here <https://zenodo.org/records/10124594>`_.

        We refere to :cite:p:`casenave2023mmgp`, Sections 4.1 and A.2 for a detailed description of the dataset.
        Some information is given in :numref:`t2d_descr`.

        .. _t2d_descr:

        .. csv-table:: Description
            :class: with-border
            :widths: 20, 80

            "Model", "2D quasistatic non-linear structural mechanics, small deformations, plane strain"
            "Constitutive law", "Nonlinear material"
            "Variability", "Mesh: nonparametrized geometry, top pressure P, 5 material parameters"
            "Meshes", "2D connected unstructured mesh, only triangles"
            "Scalars", "P, p1, p2, p3, p4, p5, max_von_mises, max_q, max_U2_top, max_sig22_top"
            "Fields", "U1, U2, q, sig11, sig12, sig22"

        The physical setting is illustrated in :numref:`t2d_phys_setting`. The boundary conditions are

        * :math:`\Gamma_{\rm top}`: imposed pressure
        * bottom line: Dirichlet 0 on y-axis
        * bottom-left point: Dirichlet 0 on x-axis


        .. _t2d_phys_setting:

        .. figure:: tensile2d_images/setting.png
            :class: with-shadow
            :width: 450px
            :align: center

            Physics setting

        An example of solution is illustrated in :numref:`t2d_sol_ex`.

        .. _t2d_sol_ex:

        .. figure:: tensile2d_images/meca_solution_examples.png
            :class: with-shadow
            :width: 600px
            :align: center

            Example of solution


    .. tab:: Machine learning problem

        The characteristics of the machine learning problem are listed in :numref:`t2d_inout`.

        .. _t2d_inout:

        .. csv-table:: ML problem description
            :class: with-border
            :widths: 20, 80

            "Inputs: scalars", "P, p1, p2, p3, p4, p5"
            "Inputs: other", "Mesh"
            "Outputs: scalars", "max_von_mises, max_q, max_U2_top, max_sig22_top"
            "Outputs: fields", "U1, U2, q, sig11, sig12, sig22"
            "Splits", "Train (8, 16, 32, 64, 125, 250 and 500 samples), Test (200 samples), Out-of-distribution (2 samples)"

        Seven training sets of different sizes are provided for different learning difficulties.


    .. tab:: Results

        Three models GCNN, MGN and MMGP are trained on the training set of size 500 in :cite:p:`casenave2023mmgp` (see Sections 3, D1 and D2).

        Detailed metrics and provided in :numref:`t2d_res1` and :numref:`t2d_res2`, from :cite:p:`casenave2023mmgp` Table 2.

        .. _t2d_res1:

        .. csv-table:: Relative RMSE for the ``Tensile2d`` dataset and considered quantities of interest (QoI) (best is bold)
            :class: with-border
            :widths: 25, 25, 25, 25

            "QoI", "GCNN", "MGN", "MMGP"
            "max_q", "1.6e-0", "**2.1e-1**", "6.6e-1"
            "max_von_mises", "4.4e-2", "5.8e-2", "**5.0e-3**"
            "max_sig22_top", "3.1e-3", "4.5e-3", "**1.7e-3**"
            "max_U2_top", "1.2e-1", "2.4e-2", "**5.0e-3**"
            "U1", "4.5e-2", "1.5e-2", "**3.4e-3**"
            "U2", "7.4e-2", "9.7e-2", "**5.5e-3**"
            "q", "1.3e-1", "1.1e-1", "**4.4e-2**"
            "sig11", "1.0e-1", "2.8e-2", "**3.7e-3**"
            "sig12", "4.5e-2", "7.5e-3", "**2.4e-3**"
            "sig22", "3.3e-2", "2.7e-2", "**1.4e-3**"

        .. _t2d_res2:

        .. csv-table:: :math:`Q^2` for the ``Tensile2d`` dataset and considered quantities of interest (QoI) (best is bold)
            :class: with-border
            :widths: 25, 25, 25, 25

            "QoI", "GCNN", "MGN", "MMGP"
            "max_q", "0.4310", "0.6400", "**0.9435**"
            "max_von_mises", "0.9245", "0.9830", "**0.9999**"
            "max_sig22_top", "0.9975", "0.9958", "**0.9993**"
            "max_U2_top", "0.9723", "0.9801", "**0.9997**"
            "U1", "0.9623", "0.9270", "**0.9997**"
            "U2", "0.9559", "0.9322", "**0.9995**"
            "q", "0.5691", "0.2626", "**0.7785**"
            "sig11", "0.9304", "0.8693", "**0.9999**"
            "sig12", "0.9617", "0.9868", "**0.9999**"
            "sig22", "0.9662", "0.9782", "**0.9999**"



        .. RRMSE
        .. GCNN MGN MMGP

        .. Tensile2d dataset
        .. qmax 1.6e-0 2.1e-1 6.6e-1
        .. vmax 4.4e-2 5.8e-2 5.0e-3
        .. σmax22 3.1e-3 4.5e-3 1.7e-3
        .. σmaxv 1.2e-1 2.4e-2 5.0e-3
        .. u 4.5e-2 1.5e-2 3.4e-3
        .. v 7.4e-2 9.7e-2 5.5e-3
        .. q 1.3e-1 1.1e-1 4.4e-2
        .. σ11 1.0e-1 2.8e-2 3.7e-3
        .. σ12 4.5e-2 7.5e-3 2.4e-3
        .. σ22 3.3e-2 2.7e-2 1.4e-3


        .. np.array(
        .. [
        .. [4.4e-2, 5.8e-2, 5.0e-3 ],
        .. [3.1e-3, 4.5e-3, 1.7e-3 ],
        .. [1.2e-1, 2.4e-2, 5.0e-3 ],
        .. [4.5e-2, 1.5e-2, 3.4e-3 ],
        .. [7.4e-2, 9.7e-2, 5.5e-3 ],
        .. [1.0e-1, 2.8e-2, 3.7e-3 ],
        .. [4.5e-2, 7.5e-3, 2.4e-3 ],
        .. [3.3e-2, 2.7e-2, 1.4e-3 ],
        .. ]
        .. )


        .. The leaderboad for dataset ``Tensile2d`` is in :numref:`t2d_ldb`.

        .. .. _t2d_ldb:

        .. .. csv-table:: Leaderboad using composite scores (without accumulated plasticity :math:`p`)
        ..     :class: with-border
        ..     :widths: 25, 25, 50
        ..     :header-rows: 1

        ..     "Rank", "Method", "Composite score"
        ..     1, "MMGP", ":math:`3.5\times 10^{-3}`"
        ..     2, "MGN", ":math:`3.3\times 10^{-2}`"
        ..     3, "GCNN", ":math:`5.8\times 10^{-2}`"


        .. .. _t2d_res_image:

        .. .. figure:: tensile2d_images/res_tensile2d.png
        ..     :class: with-shadow
        ..     :width: 800px
        ..     :align: center

        ..     Means and standard deviations (gray) of the relative RMSE and Q2 scalar regression coefficients for the ``Tensile2d`` dataset and considered quantities of interest (QoI) (best is bold)
