Tensile2d
=========


.. tabs::

    .. tab:: Dataset

        The dataset ``Tensile2d`` can be downloaded `here <https://zenodo.org/>`_.
        A description is given in :numref:`t2d_descr`.

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

            "Inputs", "Mesh, P, p1, p2, p3, p4, p5"
            "Outputs", "max_von_mises, max_q, max_U2_top, max_sig22_top, U1, U2, q, sig11, sig12, sig22"
            "Splits", "Train (500 samples), Test (200 samples), Out-of-distribution (2 samples)"

    .. tab:: Leaderboard


        The leaderboad for dataset ``Tensile2d`` is in :numref:`t2d_ldb`.

        .. _t2d_ldb:

        .. csv-table:: Leaderboad using composite scores (without accumulated plasticity :math:`p`)
            :class: with-border
            :widths: 25, 25, 50
            :header-rows: 1

            "Rank", "Method", "Composite score"
            1, "MMGP", ":math:`3.5\times 10^{-3}`"
            2, "MGN", ":math:`3.3\times 10^{-2}`"
            3, "GCNN", ":math:`5.8\times 10^{-2}`"

        Detailed metrics and provided in :numref:`t2d_res`.

        .. _t2d_res:

        .. figure:: tensile2d_images/res_tensile2d.png
            :class: with-shadow
            :width: 800px
            :align: center

            Detailed metrics from :cite:p:`casenave2023mmgp`


        .. RRMSE
        .. GCNN MGN MMGP

        .. Tensile2d dataset
        .. vmax 4.4e-2 5.8e-2 5.0e-3
        .. σmax22 3.1e-3 4.5e-3 1.7e-3
        .. σmaxv 1.2e-1 2.4e-2 5.0e-3
        .. u 4.5e-2 1.5e-2 3.4e-3
        .. v 7.4e-2 9.7e-2 5.5e-3
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
