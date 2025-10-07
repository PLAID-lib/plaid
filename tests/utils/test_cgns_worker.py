# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports


# %% Tests
class Test_cgns_worker:
    def test_import(self):
        from plaid.utils import cgns_worker

        cgns_worker.logger
