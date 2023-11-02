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

        .. csv-table:: Leaderboad using composite score
            :class: with-border
            :widths: 50, 50
            :header-rows: 1

            "Method", "Composite score"
            "MMGP", ":math:`1\times 10^{-3}`"
            "GCNN", ":math:`2\times 10^{-3}`"
            "MGN", ":math:`3\times 10^{-3}`"

        Detailed metrics and provided in :numref:`r37_res`.

        .. _r37_res:

        .. figure:: rotor37_images/res_rotor37.png
            :class: with-shadow
            :width: 800px
            :align: center

            Detailed metrics
