AirfRANS
========

.. tabs::

    .. tab:: Dataset

        The dataset ``AirfRANS`` can be downloaded `here <https://zenodo.org/>`_.
        A description is given in :numref:`arf_descr`.

        .. _arf_descr:

        .. csv-table:: Description
            :class: with-border
            :widths: 20, 80

            "Model", "3D compressible Navier-Stokes"
            "Variability", "Mesh (drawn in the NACA 4 and 5 digit series), inlet_velocity, angle_of_attack"
            "Meshes", "2D connected unstructured mesh, only triangles"
            "Scalars", "inlet_velocity, angle_of_attack, C_L, C_D"
            "Fields", "U_x, U_y, p, nu_t"

        Exemple meshes are illustrated in :numref:`arf_phys_setting`.

        .. _arf_phys_setting:

        .. figure:: airfrans_images/airfrans_mesh_example.png
            :class: with-shadow
            :width: 800px
            :align: center

            Physics setting

        Solution examples are illustrated in :numref:`arf_sol_ex`.

        .. _arf_sol_ex:

        .. figure:: airfrans_images/airfrans_solution_example.png
            :class: with-shadow
            :width: 800px
            :align: center

            Example of solution


    .. tab:: Machine learning problem

        The characteristics of the machine learning problem are listed in :numref:`arf_inout`.

        .. _arf_inout:

        .. csv-table:: ML problem description
            :class: with-border
            :widths: 20, 80

            "Inputs", "Mesh, inlet_velocity, angle_of_attack"
            "Outputs", "C_L, C_D, U_x, U_y, p, nu_t"
            "Splits", "Train (800 samples), Test (200 samples)"

    .. tab:: Leaderboard


        The leaderboad for dataset ``AirfRANS`` is in :numref:`arf_ldb`.

        .. _arf_ldb:

        .. csv-table:: Leaderboad using composite score
            :class: with-border
            :widths: 50, 50
            :header-rows: 1

            "Method", "Composite score"
            "MMGP", ":math:`1\times 10^{-3}`"
            "GCNN", ":math:`2\times 10^{-3}`"
            "MGN", ":math:`3\times 10^{-3}`"

        Detailed metrics and provided in :numref:`arf_res`.

        .. _arf_res:

        .. figure:: airfrans_images/res_airfrans.png
            :class: with-shadow
            :width: 800px
            :align: center

            Detailed metrics
