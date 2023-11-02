Rotor37
=======

.. tabs::

    .. tab:: Dataset

        The dataset ``Rotor37`` can be downloaded `here <https://zenodo.org/>`_.
        A description is given in :numref:`r37_descr`.

        .. _r37_descr:

        .. csv-table:: Description
            :class: with-border
            :widths: 20, 80

            "Model", "3D compressible Navier-Stokes"
            "Variability", "Mesh: nonparametrized geometry, input pressure P, rotation velocity omega"
            "Meshes", "2D connected unstructured mesh (in 3D ambiant space), only quads"
            "Scalars", "P, omega, in_massflow, out_massflow, compression_rate, isentropic_efficiency, polytropic_efficiency"
            "Fields", "Pressure, Temperature, Density, Energy"

        An example mesh is illustrated in :numref:`r37_mesh`.

        .. _r37_mesh:

        .. figure:: rotor37_images/rotor37_example_mesh.png
            :class: with-shadow
            :width: 800px
            :align: center

            Example mesh

        An example of solution pressure is illustrated in :numref:`r37_sol_ex`.

        .. _r37_sol_ex:

        .. figure:: rotor37_images/rotor37_example_pressure.png
            :class: with-shadow
            :width: 250px
            :align: center

            Example of solution pressure


    .. tab:: Machine learning problem

        The characteristics of the machine learning problem are listed in :numref:`r37_inout`.

        .. _r37_inout:

        .. csv-table:: ML problem description
            :class: with-border
            :widths: 20, 80

            "Inputs", "Mesh, P, omega"
            "Outputs", "in_massflow, out_massflow, compression_rate, isentropic_efficiency, polytropic_efficiency, Pressure, Temperature, Density, Energy"
            "Splits", "Train (1250/2500/5000 samples), Test (2187 samples)"

    .. tab:: Leaderboard


        The leaderboad for dataset ``Rotor37`` is in :numref:`r37_ldb`.

        .. _r37_ldb:

        .. csv-table:: Leaderboad using composite scores
            :class: with-border
            :widths: 25, 25, 50
            :header-rows: 1

            "Rank", "Method", "Composite score"
            1, "MMGP", ":math:`1.7\times 10^{-3}`"
            2, "GCNN", ":math:`6.0\times 10^{-3}`"
            3, "MGN", ":math:`9.2\times 10^{-3}`"

        Detailed metrics and provided in :numref:`r37_res`.

        .. _r37_res:

        .. figure:: rotor37_images/res_rotor37.png
            :class: with-shadow
            :width: 800px
            :align: center

            Detailed metrics from :cite:p:`casenave2023mmgp`


        .. bibliography::
            :style: unsrt


        .. RRMSE
        .. GCNN MGN MMGP

        .. Rotor37 dataset
        .. m 4.4e-3 5.4e-3 5.0e-4
        .. p 4.4e-3 5.3e-3 4.8e-4
        .. η 3.1e-3 7.2e-3 5.0e-4
        .. γ 2.9e-3 6.5e-3 4.6e-4
        .. P 1.7e-2 1.7e-2 7.2e-3
        .. T 3.9e-3 1.4e-2 8.2e-4

        .. np.array(
        .. [
        .. [4.4e-3, 5.4e-3, 5.0e-4],
        .. [4.4e-3, 5.3e-3, 4.8e-4],
        .. [3.1e-3, 7.2e-3, 5.0e-4],
        .. [2.9e-3, 6.5e-3, 4.6e-4],
        .. [1.7e-2, 1.7e-2, 7.2e-3],
        .. [3.9e-3, 1.4e-2, 8.2e-4],
        .. ]
        .. }
