from sklearn.base import BaseEstimator, OutlierMixin
import numpy as np


class EnetConexHull(BaseEstimator, OutlierMixin):
    """
    One-Class Classifier for Anomaly Detection using Elastic Net and Convex Hull.

    Parameters
    ----------
    landa1 : float (default=0.5)
        The weight of L1 regularization. Must be in the range [0,1].
    target : int (default=1)
        The label of the target class (e.g., 1 for inliers)
    lb : ndarray of shape (n_samples, ) or None 
        Lower bound for optimization variables. Defaults to zeros
    solver : str, callable (default='cvxopt')
        The solver used for quadratic programming optimization.
    metric : str, callable
        The Kernel metric used for pairwise similarity computation
    only_target : bool (default=True)
        Whether to use only the target class samples for training.
    thr : float (default=1.0)
        The decision threshold for classifying anomalies.   
        

    Notes
    -----
    Why use BaseEstimator and OutlierMixin?
    - **BaseEstimator**:
        - Automatically provides `get_params` and `set_params` for parameter management.
        - Ensure compatibility with Scikit-learn tools like `Pipeline` and `GridSearchCV`.
        - Reduces repetitive code by managing parameters automatically.
    - **OutlierMixin**:
        - Adds specific functionality for anomaly detection (e.g., `fit_predict`).
        - Identifies the class as an anomaly detection model within Scikit-learn's ecosystem.
        - Simplifies integration with Scikit-learn's evaluation tools.
    - Implements support for `GridSearchCV` and `Pipeline` by providing compatible `get_params` and `set_params` methods.
    - Designed to work seamlessly with Scikit-learn's tools and standards for hyperparameter tuning and model chaining.
    """

    def __init__(self, landa1=0.5, target=1, lb=None, solver='cvxopt',
                 metric=None, only_target=True, thr=1.0):
        self.landa1      = landa1
        self.target      = target
        self.lb          = lb
        self.solver      = solver
        self.metric      = metric
        self.only_target = only_target
        self.thr         = thr