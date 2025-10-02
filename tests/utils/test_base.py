# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

import pytest

from plaid.utils import base


# %% Tests
class Test_base:
    def test_generate_random_ASCII(self):
        base.generate_random_ASCII()

    def test_safe_len(self):
        assert base.safe_len([0,1]) == 2
        assert base.safe_len(0) == 0

    def test_get_mem(self):
        base.get_mem()
