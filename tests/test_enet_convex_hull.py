from models import EnetConvexHull
import pytest
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels, linear_kernel
from sklearn.utils.validation import check_random_state

def test_validate_params():
    """
    Test the `_validate_params` method with valid and invalid parameters.
    """

    # Valid parameters
    model = EnetConvexHull(landa1=0.5, target=1, thr=0.1)
    try:
        model._validate_params()
    except ValueError as e:
        assert False, f"`_validate_params` raised an exception with valid parameters: {e}"
    
    # Invalid landa1 (out of range)
    model = EnetConvexHull(landa1=1.5, target=1, thr=1.0)
    try:
        model._validate_params()
        assert False, "Expected ValueError for `landa1` out of range, but none was raised."
    except ValueError as e:
        assert "landa1" in str(e), "Expected ValueError message to include `landa1`."

    # Invalid target (non-integer)
    model = EnetConvexHull(landa1=0.5, target='invalid', thr=1.0)
    try:
        model._validate_params()
        assert False, "Expected ValueError for non-integer `target`, but none was raised."
    except ValueError as e:
        assert "target" in str(e), "Expected ValueError message to include `target`"

    # Invalid thr (negative value)
    model = EnetConvexHull(landa1=0.5, target=1, thr=-0.5)
    try:
        model._validate_params()
        assert False, "Expected ValueError for negative `thr`, but none was raised."
    except ValueError as e:
        assert "thr" in str(e), "Expected ValueError message to include `thr`."

def test_pairwise_kernels_similarity():
    """
    Test the `pairwise_kernels_similarity` method with different inputs.
    """
    model = EnetConvexHull(metric='linear')

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
    model = EnetConvexHull(metric='linear')

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
    unfitted_model = EnetConvexHull(metric='linear')
    try:
        unfitted_model.__calculate_z__(sample)
        assert False, "Expected an error when `__calculate_z__` is called before fit, but none was raised."
    except ValueError as e:
        assert "not fitted" in str(e), "Error message for calling `__calcualte_z__` on unfitted model is incorrect."


def test_predict():
    """
    Test the predict method of EnetConvexHull.
    """
    model = EnetConvexHull(metric='linear', thr=1.0)

    X_train = np.array([[1,2], [3,4], [5, 6]])
    model.fit(X_train)

    X_test = np.array([[1,2], [7, 8], [10, 12]])

    try:
        predictions = model.predict(X_test)
        assert predictions.shape == (X_test.shape[0], ), "The shape of predictions is incorrect."
        assert np.all(np.isin(predictions, [1,-1])), "Predictions should only contain 1 and -1."
    except Exception as e:
        assert False, f"prdict raised an error: {e}"

    unfitted_model = EnetConvexHull(metric='linear')
    try:
        unfitted_model.predict(X_test)
        assert False, "Expected and error when predict is called before `fit`, but none was raised."
    except ValueError as e:
        assert "not fitted" in str(e), "Error message for calling predict on unfitted model is incorrect."


def test_kernel_params():
    """
    Test kernel_params functionality in EnetConvexHull class.
    """
    
    # Initialize the model with specific kernel parameters
    model = EnetConvexHull(metric='rbf', kernel_params={'gamma': 0.1})

    # Input data
    X = np.array([[1,2], [3,4], [5,6]])

    # Validate the initial kernel_params
    assert model.kernel_params == {'gamma':0.1}, "Initial kernel_params are not set correctly."

    # Update kernel_params
    model.set_kernel_params(gamma=0.2)
    assert model.kernel_params == {'gamma':0.2}, "Failed to update kernel_params."

    # Validate the kernel computation with updated kernel_params
    expected_result = pairwise_kernels(X, metric='rbf', gamma=0.2)
    result = pairwise_kernels(X, metric=model.metric, **model.kernel_params)

    assert np.allclose(result, expected_result), "Kernel computation with updated kernel_params is incorrect."


def test_adjust_kernel():
    """
    Test the `_adjust_kernel` method of  EnetConvexHull for correctness.
    """
    # Generate mock data
    rng = check_random_state(42)
    X   = rng.rand(5, 3) # 5 samples, 3 features

    # Initialize the model
    model = EnetConvexHull(metric='linear', kernel_params={})

    # Compute the adjusted kernel matrix
    adjusted_kernel = model._adjust_kernel(X)

    assert adjusted_kernel.shape == (5, 5), "Adjusted kernel matrix must be square with shape (n_samples, n_samples)"
    assert np.allclose(adjusted_kernel.sum(axis=0), adjusted_kernel.sum(axis=1)), (
        "Adjusted kernel matrix should be symmetric"
    )
    assert isinstance(adjusted_kernel, np.ndarray), "Adjusted kernel matrix must be a numpy array."


import pytest

def test_validate_kernel_params():
    """
    Test the _validate_kernel_params method of EnetConvexHull.
    """

    # Case 1: Valid parameters for 'rbf' kernel
    model_rbf = EnetConvexHull(metric='rbf', kernel_params={'gamma': 0.5})
    try:
        model_rbf._validate_kernel_params()
    except ValueError:
        pytest.fail("Validation failed for valid 'rbf' kernel parameters.")

    # Case 2: Valid parameters for 'poly' kernel
    model_poly = EnetConvexHull(metric='poly', kernel_params={'gamma': 0.3, 'degree': 3, 'coef0': 1.0})
    try:
        model_poly._validate_kernel_params()
    except ValueError:
        pytest.fail("Validation failed for valid 'poly' kernel parameters.")

    # Case 3: Valid parameters for 'sigmoid' kernel
    model_sigmoid = EnetConvexHull(metric='sigmoid', kernel_params={'gamma': 0.2, 'coef0': 1.5})
    try:
        model_sigmoid._validate_kernel_params()
    except ValueError:
        pytest.fail("Validation failed for valid 'sigmoid' kernel parameters.")

    # Case 4: Invalid gamma for 'rbf'
    model_invalid_gamma = EnetConvexHull(metric='rbf', kernel_params={'gamma': 'invalid'})
    with pytest.raises(ValueError, match="Invalid gamma value: invalid.*"):
        model_invalid_gamma._validate_kernel_params()

    # Case 5: Negative degree for 'poly'
    model_negative_degree = EnetConvexHull(metric='poly', kernel_params={'gamma': 0.3, 'degree': -1, 'coef0': 1.0})
    with pytest.raises(ValueError, match="Invalid degree value: -1.*"):
        model_negative_degree._validate_kernel_params()

    # Case 6: Invalid coef0 for 'sigmoid'
    model_invalid_coef0 = EnetConvexHull(metric='sigmoid', kernel_params={'gamma': 0.2, 'coef0': 'invalid'})
    with pytest.raises(ValueError, match="Invalid coef0 value: invalid.*"):
        model_invalid_coef0._validate_kernel_params()

    # Case 7: Missing metric
    model_no_metric = EnetConvexHull(kernel_params={'gamma': 0.3})
    with pytest.raises(ValueError, match="The kernel metric must be specified."):
        model_no_metric._validate_kernel_params()


