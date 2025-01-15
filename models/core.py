from sklearn.base             import BaseEstimator, OutlierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.metrics.pairwise import pairwise_kernels
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
        self.landa1       = landa1
        self.landa2       = 1 - self.landa1
        self.target       = target
        self.lb           = lb
        self.solver       = solver
        self.metric       = metric
        self.only_target  = only_target
        self.thr          = thr
        self.return_label = False

    def _validate_params(self):
        """
        Validate the input parameters for the EnetConvexHull model.

        Raises
        ------
        ValueError
            If any parameter is invalid.
        """

        # landa1 must be in [0, 1]
        if not isinstance(self.landa1, (int, float)) or not (0 <= self.landa1 <= 1):
            raise ValueError(f"landa1 ({self.landa1}) must be a float in the range [0,1].")
        
        # target must be an integer
        if not isinstance(self.target, int):
            raise ValueError(f"target ({self.target}) must be an integer.")
        
        # lb must be None or a numpy array
        if self.b is not None and not isinstance(self.lb, np.ndarray):
            raise ValueError(f"lb must be a numpy array or None. Got {type(self.lb)} instead.")

        # solver must be a string or callable
        if not (self.solver is None or isinstance(self.solver, str) or callable(self.solver)):
            raise ValueError(f"solver ({self.solver}) must be a string, callable or None.")

        # metric must be a string or callable
        if not (self.metric is None or isinstance(self.metric, str) or callable(self.metric)):
            raise ValueError(f"metric ({self.metric}) must be a string, callable or None.")
        
        # only_target must be a boolean
        if not isinstance(self.only_target, bool):
            raise ValueError(f"only_target ({self.only_target}) must be a boolean.")
        
        # thr must be a positive float
        if not isinstance(self.thr, (int, float)) or self.thr <= 0:
            raise ValueError(f"thr ({self.thr}) must be a positive float.")
        
    def fit(self, X, y=None):
        """
        Fit the EnetConvexHull model to the given data.


        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples, )
            Class labels. If provided, only samples with `target` label will be used.

        Returns
        -------
        self : object 
            Fitted instance of the model.
        """

        # Validate parameters and inputs
        self._validate_params()
        X, y = check_X_y(X, y, accept_sparse=False, ensure_2d=True, dtype=np.float64)

        # Select target samples if applicable
        if y is not None:
            self.return_label = True
            # sklearn ``metrics`` API needs attribute ``classes_``
            self.classes_ = np.unique(y) # Required for scikit-learn compatibility
            mask = (y == self.target)
            self.X_target = X[mask, :] if self.only_target else X
        else:
            self.X_target = X

        # Compute the kernel matrix
        self.G = pairwise_kernels(self.X_target, metric=self.metric)
        self.G_sum = np.sum(self.G)
        self.n, self.m = self.X_target.shape

        # Compute the optimization matrix
        BTB = self.pairwise_kernels_similarity(self.X_target, metric=self.metric)
        self.P = (self.landa2 * np.identity(self.n)) + BTB

        # Inirialize lower bounds
        if self.lb is None:
            self.lb = np.zeros((self.n, 1))

        # Store additional parameters for later use. # for scikit-learn compatibility
        self.is_fitted_ = True

        return self

    def pairwise_kernels_similarity(self, X, Y=None, metric=None ):
        """
        Compute the adjusted pairwise kernel similarity matrix.

        Parameters
        ----------
        X : ndarray of shape (n_smaples_X, n_features)
            Input data for the first set.
        Y : ndarray of shape (n_samples_Y, n_features)
            Input data for the second set. If None, Y is set to X.
        metric : str, callable, optional (Defaults = the model's metric)
            Kernel metric to use. 

        Returns
        -------
        similarity_matrix : ndarray of shape (n_samples_X, n_sample_Y)
            Adjusted pairwise kernel similarity matrix.
        """

        if metric is None:
            metric = self.metric
        if Y is None:
            Y = X
        
        # Compute the pairwise kernel matrix
        G = pairwise_kernels(X, Y, metric=metric)

        # Compute adjustments for the kernel
        n, m = X.shape[0], Y.shape[0]
        G_kl = np.sum(G)
        Grow  = np.sum(G, axis=1) # row-wise sum #G[i,:]  foreach i
        Gcol  = np.sum(G, axis=0) # column-wise sum #G[:,j] foreach j
        
        # Broadcasting sums to match matrix dimensions
        Grow  = np.broadcast_to(Grow, shape=(n, n)).T
        Gcol = np.broadcast_to(Gcol,shape=(m,m))

        g = G - ( (Grow+Gcol)/n ) + (1/n**2)*G_kl
        #for i in tqdm(range(n),desc='BTB',leave=False):
        #    for j in range(m):
        #        g[i,j] = G[i,j] - (1/n)*( (np.sum(G[i,:])) + np.sum(G[:, j]) ) + (1/n**2)*G_kl
        
        return g # g is (n_sample_x, n_sample_y) matrix


    def predict(self, X):
        """
        Predict whether a sample is an inlier or outlier.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.
        
        Returns
        -------
        predictions : ndarray of shape (n_samples)
            Prediction 1 for inliers, -1 for outliers.
        """
        # Use `decision_function` to compute scores.
        scores = self.decision_function(X)

        # Classifiy based on threshold
        predictions = np.where(scores <= self.thr, 1, -1)

        return predictions

    def decision_function(self, X):
        """
        Compute anomaly scores for each sample

        Parameters
        ----------
        X : ndarray of shape (n_sample, n_features)
            Input data.

        Returns
        -------
        scores : ndarray of shape (n_samples, n_features)
            Anomaly scores for each sample. Lower scores indicate closer proximity
            to the target class
        """

        # Check if the model is fitted
        if not hasattr(self, "is_fitted_"):
            raise ValueError(f"This {self.__class__.__name__} instance is not fitted yet. Call `fit` before using this method.")
        
        # Validate input data
        X = check_array(X, ensure_2d=True, dtype=np.float64)

        # Compute decision scores
        scores = np.array([self.__calculate_z__(sample.reshape(1, -1)) for sample in X])
        return scores
    def __calculate_z__(self,X):
        NotImplemented