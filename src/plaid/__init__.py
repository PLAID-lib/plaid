# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

try:
    from ._version import __version__
except ImportError:
    __version__ = "None"

__all__ = ["__version__"]