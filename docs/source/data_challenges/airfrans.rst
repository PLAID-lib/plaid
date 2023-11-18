AirfRANS
========

.. tabs::

    .. tab:: Dataset

        .. The dataset ``AirfRANS`` can be downloaded `here <https://zenodo.org/>`_. There is a standard version and a remeshed (coarsened) one.

        The dataset ``AirfRANS`` will soon be available on `Zenodo <https://zenodo.org/>`_.

        We refere to :cite:p:`airfrans` and :cite:p:`casenave2023mmgp`, Sections 4.1 for a detailed description of the dataset.
        Some information is given in :numref:`arf_descr`.

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

            "Inputs: scalars", "inlet_velocity, angle_of_attack"
            "Inputs: other", "Mesh"
            "Outputs: scalars", "C_L, C_D"
            "Outputs: fields", "U_x, U_y, p, nu_t"
            "Splits", "Scarse train (200 samples), Train (800 samples), Test (200 samples)"


    .. tab:: Results

        In :cite:p:`casenave2023mmgp` (see Sections 3, D1 and D2), two models GCNN and MGN
        are trained in the remeshed version and MMGP is trained on the standard version, all on the training
        set of size 800.

        Detailed metrics and provided in :numref:`arf_res1` and :numref:`arf_res2`, from :cite:p:`casenave2023mmgp` Table 2.

        .. _arf_res1:

        .. csv-table:: Relative RMSE for the ``AirfRANS`` dataset and considered quantities of interest (QoI) (best is bold)
            :class: with-border
            :widths: 25, 25, 25, 25

            "QoI", "GCNN", "MGN", "MMGP"
            "C_D", "6.1e-2", "4.9e-2", "**3.3e-2**"
            "C_L", "4.1e-1", "2.4e-1", "**8.0e-3**"
            "U_x", "5.6e-2", "8.3e-2", "**1.8e-2**"
            "U_y", "4.2e-2", "1.2e-1", "**1.5e-2**"
            "p", "8.5e-2", "9.9e-2", "**5.1e-2**"


        .. _arf_res2:

        .. csv-table:: :math:`Q^2` for the ``AirfRANS`` dataset and considered quantities of interest (QoI) (best is bold)
            :class: with-border
            :widths: 25, 25, 25, 25

            "QoI", "GCNN", "MGN", "MMGP"
            "C_D", "0.9596", "0.9743", "**0.9831**"
            "C_L", "0.9776", "0.9851", "**0.9999**"
            "U_x", "0.9659", "0.9110", "**0.9749**"
            "U_y", "0.9683", "0.7516", "**0.9806**"
            "p", "0.9602", "0.9390", "**0.9934**"


        .. RRMSE
        .. GCNN MGN MMGP

        .. AirfRANS dataset
        .. CD 6.1e-2 4.9e-2 3.3e-2
        .. CL 4.1e-1 2.4e-1 8.0e-3
        .. U_x 5.6e-2 8.3e-2 1.8e-2
        .. U_y 4.2e-2 1.2e-1 1.5e-2
        .. p 8.5e-2 9.9e-2 5.1e-2


        .. np.array(
        .. [
        .. [6.1e-2, 4.9e-2, 3.3e-2 ],
        .. [4.1e-1, 2.4e-1, 8.0e-3 ],
        .. [5.6e-2, 8.3e-2, 1.8e-2 ],
        .. [4.2e-2, 1.2e-1, 1.5e-2 ],
        .. [8.5e-2, 9.9e-2, 5.1e-2 ],
        .. ]
        .. )

        .. The leaderboad for dataset ``AirfRANS`` is in :numref:`arf_ldb`.

        .. .. _arf_ldb:

        .. .. csv-table:: Leaderboad using composite scores (without field :math:`\nu_t`)
        ..     :class: with-border
        ..     :widths: 25, 25, 50
        ..     :header-rows: 1

        ..     "Rank", "Method", "Composite score"
        ..     1, "MMGP", ":math:`2.5\times 10^{-2}`"
        ..     2, "MGN", ":math:`1.2\times 10^{-1}`"
        ..     3, "GCNN", ":math:`1.3\times 10^{-1}`"

        .. Detailed metrics and provided in :numref:`arf_res`.

        .. .. _arf_res:

        .. .. figure:: airfrans_images/res_airfrans.png
        ..     :class: with-shadow
        ..     :width: 800px
        ..     :align: center

        ..     Detailed metrics from :cite:p:`casenave2023mmgp`
