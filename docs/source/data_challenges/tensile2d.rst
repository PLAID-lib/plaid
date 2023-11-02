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

        .. csv-table:: Leaderboad using composite score
            :class: with-border
            :widths: 50, 50
            :header-rows: 1

            "Method", "Composite score"
            "MMGP", ":math:`1\times 10^{-3}`"
            "GCNN", ":math:`2\times 10^{-3}`"
            "MGN", ":math:`3\times 10^{-3}`"

        Detailed metrics and provided in :numref:`t2d_res`.

        .. _t2d_res:

        .. figure:: tensile2d_images/res_tensile2d.png
            :class: with-shadow
            :width: 800px
            :align: center

            Detailed metrics
