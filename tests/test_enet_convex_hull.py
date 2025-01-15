from models import EnetConexHull
import pytest


def test_validate_params():
    """
    Test the `_validate_params` method with valid and invalid parameters.
    """

    # Valid parameters
    model = EnetConexHull(landa1=0.5, target=1, thr=0.1)
    try:
        model._validate_params()
    except ValueError as e:
        assert False, f"`_validate_params` raised an exception with valid parameters: {e}"
    
    # Invalid landa1 (out of range)
    model = EnetConexHull(landa1=1.5, target=1, thr=1.0)
    try:
        model._validate_params()
        assert False, "Expected ValueError for `landa1` out of range, but none was raised."
    except ValueError as e:
        assert "landa1" in str(e), "Expected ValueError message to include `landa1`."

    # Invalid target (non-integer)
    model = EnetConexHull(landa1=0.5, target='invalid', thr=1.0)
    try:
        model._validate_params()
        assert False, "Expected ValueError for non-integer `target`, but none was raised."
    except ValueError as e:
        assert "target" in str(e), "Expected ValueError message to include `target`"

    # Invalid thr (negative value)
    model = EnetConexHull(landa1=0.5, target=1, thr=-0.5)
    try:
        model._validate_params()
        assert False, "Expected ValueError for negative `thr`, but none was raised."
    except ValueError as e:
        assert "thr" in str(e), "Expected ValueError message to include `thr`."