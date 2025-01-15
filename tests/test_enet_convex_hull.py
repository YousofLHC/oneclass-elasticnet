from models import EnetConexHull
import pytest


def test_initialization():
    """
    Test initialization and parameter validation.
    """

    # Valid initialization
    model = EnetConexHull(landa1=0.5, target=1, thr=1.0)
    assert model.landa1 == 0.5, "`landa1` should be initialized to `0.5`"
    assert model.target == 1  , "`target` should be initialized to `1`"
    assert model.thr    == 1.0, "`thr`    should be initialized to `1.0`"   

    # Invalid initialization: landa1 out of range
    with pytest.raises(ValueError) as excinfo:
        EnetConexHull(landa1=-0.1)
    assert "landa1" in str(excinfo.value), "Exception should indicate invalid `landa1` value."

    # Invalid initialization: negative thr
    with pytest.raises(ValueError) as excinfo:
        EnetConexHull(thr=-1.0)
    assert "thr" in str(excinfo.value), "Exception should indicate invalid `thr` value."