"""Examples for PLAID objects."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

_HF_REPOS = {
'vki_ls59':'PLAID-datasets/VKI-LS59',
'elastoplastodynamics':'PLAID-datasets/2D_ElastoPlastoDynamics',
'multiscale_hyperelasticity':'PLAID-datasets/2D_Multiscale_Hyperelasticity',
'tensile2d':'PLAID-datasets/Tensile2d',
'rotor37':'PLAID-datasets/Rotor37',
'profile2d':'PLAID-datasets/2D_profile',
'airfrans_clipped':'PLAID-datasets/AirfRANS_clipped',
'airfrans_original':'PLAID-datasets/AirfRANS_original',
'airfrans_remeshed':'PLAID-datasets/AirfRANS_remeshed',
}


AVAILABLE_EXAMPLES = list(_HF_REPOS.keys())


from .dataset import datasets
from .sample import samples

__all__ = ["datasets", "samples"]
