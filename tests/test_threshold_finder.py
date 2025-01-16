import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from qpsolvers import cvxopt_solve_qp
from models import EnetConexHull
from models import ThresholdFinder

def test_threshold_finder():
    """
    Test the ThresholdFinder class for functionality and correctness.
    """
    # Initialize sample data
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([1, 1, 1, 1])  # All target samples
    
    # Initialize the EnetConexHull model
    model = EnetConexHull(landa1=0.5, metric='linear', solver='cvxopt')
    model.fit(X, y)

    # Initialize ThresholdFinder with the model
    finder = ThresholdFinder(model=model, X=X)

    # Find z values and maximum z
    z, max_z = finder.find(outs='max')

    # Assertions
    assert z.shape == (X.shape[0], 1), "The shape of z is incorrect."
    assert np.isclose(max_z, np.max(z)), "The maximum z value is incorrect."
    assert np.all(z >= 0), "Some z values are negative, which is unexpected."

    print("ThresholdFinder test passed successfully.")

if __name__ == "__main__":
    test_threshold_finder()