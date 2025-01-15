from models import EnetConexHull
import pytest
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels

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

def test_pairwise_kernels_similarity():
    """
    Test the `pairwise_kernels_similarity` method with different inputs.
    """
    model = EnetConexHull(metric='linear')

    # Input data
    X = np.array([[1,2], [3,4], [5,6]])
    Y = np.array([[1,2],[7,8]])

    # Compute similarity with default metric ('linear')
    model.metric='linear'
    similarity_matrix = model.pairwise_kernels_similarity(X, Y)
    assert similarity_matrix.shape == (3,2), "The similarity matrix shape is incorrect."
    #assert np.allclose(similarity_matrix, pairwise_kernels(X, Y, metric='linear')), (
    #    "The similarity matrix values do not match the expected output."
    #)

    # Test with Y=None (self-similarity)
    similarity_matrix_self = model.pairwise_kernels_similarity(X)
    assert similarity_matrix_self.shape == (3, 3), (
        "The self-similarity matrix is incorrect when Y is None."
    )
    #assert np.allclose(similarity_matrix_self, pairwise_kernels(X, X, metric='linear')), (
    #    "The self-similarity matrix values do not match the expected output."
    #)

    # Test with a different metric ('rbf')
    model.metric = 'rbf'
    similarity_matrix_rbf = model.pairwise_kernels_similarity(X, Y)
    assert similarity_matrix_rbf.shape == (3,2), "The similarity matrix shape is incorrect."
    #assert np.allclose(similarity_matrix_rbf, pairwise_kernels(X, Y, metric='rbf')), (
    #    "The `RBF` similarity matrix values do not match the expected output."
    #)

    # Test with invalid metric
    model.metric='invalid_metric'
    #try:
    #    model.pairwise_kernels_similarity(X, Y)
    #    assert False, "Expected an error with an invalid kernel metric, but none was raised."
    #except ValueError as e:
    #    assert "'invalid_metric' instead" in str(e), "Error message for invalid metric is incorrect."
    with pytest.raises(ValueError, match="'invalid_metric' instead"):
        model.pairwise_kernels_similarity(X, Y)

def test_calculate_z():
    """
    Test the __calculate_z__ method
    """
    model = EnetConexHull(metric='linear')

    # Sample training data
    X_train = np.array([[1,2], [3,4], [5,6]])
    model.fit(X_train)

    # Sample input for calculation
    sample = np.array([[2,3]])

    # Call __calculate_z__
    try:
        z_value = model.__calculate_z__(sample)
        assert isinstance(z_value, float), "The z-value should be a float."
    except Exception as e:
        assert False, f"__calculate_z__ raised an error: {e}"

    # Test without fitting the model
    unfitted_model = EnetConexHull(metric='linear')
    try:
        unfitted_model.__calculate_z__(sample)
        assert False, "Expected an error when `__calculate_z__` is called before fit, but none was raised."
    except ValueError as e:
        assert "not fitted" in str(e), "Error message for calling `__calcualte_z__` on unfitted model is incorrect."


def test_predict():
    """
    Test the predict method of EnetConexHull.
    """
    model = EnetConexHull(metric='linear', thr=1.0)

    X_train = np.array([[1,2], [3,4], [5, 6]])
    model.fit(X_train)

    X_test = np.array([[1,2], [7, 8], [10, 12]])

    try:
        predictions = model.predict(X_test)
        assert predictions.shape == (X_test.shape[0], ), "The shape of predictions is incorrect."
        assert np.all(np.isin(predictions, [1,-1])), "Predictions should only contain 1 and -1."
    except Exception as e:
        assert False, f"prdict raised an error: {e}"

    unfitted_model = EnetConexHull(metric='linear')
    try:
        unfitted_model.predict(X_test)
        assert False, "Expected and error when predict is called before `fit`, but none was raised."
    except ValueError as e:
        assert "not fitted" in str(e), "Error message for calling predict on unfitted model is incorrect."