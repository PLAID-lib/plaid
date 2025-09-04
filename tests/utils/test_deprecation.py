# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

import warnings

import pytest

import plaid.utils.deprecation as dep

# %% Tests


def test_deprecated_function_warns():
    @dep.deprecated("use new_func instead", version="1.0", removal="2.0")
    def old_func(x):
        return x * 2

    # Capture warning
    with pytest.warns(DeprecationWarning, match="use new_func instead"):
        result = old_func(5)

    assert result == 10


def test_deprecated_class_warns():
    @dep.deprecated("use NewClass instead", version="1.1")
    class OldClass:
        def __init__(self, val):
            self.val = val

    with pytest.warns(DeprecationWarning, match="use NewClass instead"):
        obj = OldClass(42)

    assert obj.val == 42


def test_deprecated_invalid_type():
    decorator = dep.deprecated("invalid use")
    with pytest.raises(
        TypeError, match="@deprecated can only be applied to functions or classes"
    ):
        decorator(pytest)
    with pytest.raises(
        TypeError, match="@deprecated can only be applied to functions or classes"
    ):
        decorator(3)
    with pytest.raises(
        TypeError, match="@deprecated can only be applied to functions or classes"
    ):
        decorator(3.14)
    with pytest.raises(
        TypeError, match="@deprecated can only be applied to functions or classes"
    ):
        decorator("test")


def test_deprecated_argument_warns_and_converts():
    @dep.deprecated_argument(
        "old", "new", converter=lambda v: v + 1, version="1.2", removal="2.0"
    )
    def func(new):
        return new * 2

    # Using old argument -> should warn and convert
    with pytest.warns(
        DeprecationWarning, match="Argument `old` is deprecated, use `new` instead."
    ):
        result = func(old=3)
    assert result == 8  # (3+1)*2

    # Using new argument directly -> no warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result2 = func(new=5)
    assert result2 == 10
    assert len(w) == 0


def test_deprecated_argument_no_old_arg():
    """Calling a function without the deprecated argument should just work."""

    @dep.deprecated_argument("old", "new")
    def func(new):
        return new * 3

    result = func(new=4)
    assert result == 12


def test_deprecated_fallback(monkeypatch):
    """Simulate Python < 3.13 where warnings.deprecated does not exist."""

    monkeypatch.setattr(dep, "deprecated_builtin", None)

    @dep.deprecated("use fallback", version="9.9", removal="10.0")
    def legacy(x):
        return x + 1

    with pytest.warns(DeprecationWarning, match="use fallback"):
        assert legacy(4) == 5

    @dep.deprecated("old class fallback")
    class LegacyClass:
        def __init__(self, x):
            self.x = x

    with pytest.warns(DeprecationWarning, match="old class fallback"):
        obj = LegacyClass(7)
    assert obj.x == 7

    @dep.deprecated_argument("old", "new", version="9.9", removal="10.0")
    def func(new):
        return new

    with pytest.warns(DeprecationWarning, match="Argument `old` is deprecated"):
        assert func(old=123) == 123


def test_deprecated_argument_converter_identity():
    """Ensure converter default (identity) is applied."""

    @dep.deprecated_argument("x", "y")
    def func(y):
        return y

    with pytest.warns(DeprecationWarning, match="Argument `x` is deprecated"):
        assert func(x="hello") == "hello"
