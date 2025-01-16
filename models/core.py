from sklearn.base             import BaseEstimator, OutlierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.metrics.pairwise import pairwise_kernels
from   qpsolvers              import solve_qp
from   tqdm                   import tqdm
import numpy as np


def ensure_fitted(func):
    """
    Decorator to ensure the model is fitted before method execution.
    """
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "is_fitted_"):
            raise ValueError(f"This {self.__class__.__name__} instance is not fitted yet. Call `fit` before using this method.")
        return func(self, *args, **kwargs)

    return wrapper

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
        Lower bound for QP optimization variables. Defaults to zeros
    solver : str, callable (default='cvxopt')
        The solver used for quadratic programming optimization.
    metric : str, callable
        The Kernel metric used for pairwise similarity computation
    only_target : bool (default=True)
        Whether to use only the target class samples for training.
    thr : float (default=1.0)
        The decision threshold for classifying anomalies.  
    kernel_params : dict (default=None)
        Additional parameters for kernel computation
    degree : int (default=3)
        Degree for poly kernels.
    gamma : {'scale', 'auto'}, float (default='scale')
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
    coef0 : float(default=1.0)
        Independent term for 'poly' and 'sigmoid' kernels. 
        

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
                 metric=None, only_target=True, thr=1.0, kernel_params=None,
                 degree=3, gamma='scale', coef0=1.0):
        self.landa1        = landa1
        self.landa2        = 1 - self.landa1
        self.target        = target
        self.lb            = lb
        self.solver        = solver
        self.metric        = metric
        self.only_target   = only_target
        self.thr           = thr
        self.return_label  = False

        # Initialize kernel parameters
        self.kernel_params = kernel_params if kernel_params else {}

        # Define kernel-specific parameters based on the metric
        if self.metric in {'poly', 'rbf', 'sigmoid'} and 'gamma' not in self.kernel_params:
            self.kernel_params['gamma'] = gamma
        if self.metric == 'poly':
            if 'degree' not in self.kernel_params:
                self.kernel_params['degree'] = degree
            if 'coef0' not in self.kernel_params:
             self.kernel_params['coef0']  = coef0
        if 'coef0' not in self.kernel_params and self.metric == 'sigmoid':
            self.kernel_params['coef0'] = coef0

    def _validate_params(self):
        """
        Validate the input parameters for the EnetConvexHull model.

        Raises
        ------
        ValueError
            If any parameter is invalid.
        """

        if not isinstance(self.landa1, (int, float)) or not (0 <= self.landa1 <= 1):
            raise ValueError(f"landa1 ({self.landa1}) must be a float in the range [0,1].")
        
        if not isinstance(self.target, int):
            raise ValueError(f"target ({self.target}) must be an integer.")
        
        # lb must be None or a numpy array
        if self.lb is not None and not isinstance(self.lb, np.ndarray):
            raise ValueError(f"lb must be a numpy array or None. Got {type(self.lb)} instead.")

        if not (self.solver is None or isinstance(self.solver, (str, callable))):
            raise ValueError(f"solver ({self.solver}) must be a string, callable or None.")

        # metric must be a string or callable
        if not (self.metric is None or isinstance(self.metric, (str, callable))):
            raise ValueError(f"metric ({self.metric}) must be a string, callable or None.")
        
        if not isinstance(self.only_target, bool):
            raise ValueError(f"only_target ({self.only_target}) must be a boolean.")
        
        # thr must be a positive float
        if not isinstance(self.thr, (int, float)) or self.thr <= 0:
            raise ValueError(f"thr ({self.thr}) must be a positive float.")

    def _validate_kernel_params(self):
        """
        Validate kernel-specific parameters based on the selected kernel metric.

        Raises
        ------
        ValueError
            If any kernel parameter is invalid or missing for the selected metric.
        """
        # Check if metric is specified
        if self.metric is None:
            raise ValueError("The kernel metric must be specified.")

        # Validate gamma
        if self.metric in {'poly', 'rbf', 'sigmoid'}:
            gamma = self.kernel_params.get('gamma', 'scale')
            if not (isinstance(gamma, (int, float)) or gamma in {'scale', 'auto'}):
                raise ValueError(f"Invalid gamma value: {gamma}. It must be 'scale', 'auto', or a positive number.")

        # Validate degree for poly kernel
        if self.metric == 'poly':
            degree = self.kernel_params.get('degree', None)
            if degree is None:
                raise ValueError("Invalid degree. The 'degree' parameter is required for the 'poly' kernel.")
            if not isinstance(degree, int) or degree <= 0:
                raise ValueError(f"Invalid degree value: {degree}. It must be a positive integer.")


        # Validate coef0 for poly and sigmoid kernels
        if self.metric in {'poly', 'sigmoid'}:
            coef0 = self.kernel_params.get('coef0', None)
            if coef0 is not None and not isinstance(coef0, (int, float)):
                raise ValueError(f"Invalid coef0 value: {coef0}. It must be a number.")

        # Additional kernel-specific validations can be added here as needed


    def _adjust_kernel(self, X, Y=None):
        """
        Compute the adjusted pairwise kernel similarity matrix.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Inpud data for the first set.
        Y : ndarray of shape (n_samples_Y, n_features)
            Input data for the second set. If None, Y is set to X.

        Returns
        -------
        adjusted_kernel : ndarray of shape (n_samples_X, n_samples_Y)
            Adjusted pairwise kernel similarity matrix.
        """
        if Y is None:
            Y = X
        G = pairwise_kernels(X, Y, metric=self.metric, **self.kernel_params)
        G_sum = G.sum()
        row_sum = G.sum(axis=1, keepdims=True)
        col_sum = G.sum(axis=0, keepdims=True)
        return G -(row_sum+col_sum)/X.shape[0] + G_sum / (X.shape[0]**2)
    def _calculate_P(self, X):
            BTB = self._adjust_kernel(X)
            P   = (self.landa2 * np.identity(X.shape[0])) + BTB
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
        self._validate_kernel_params()  # Validate kernel parameters
        if y is not None:
            X, y     = check_X_y(X, y, accept_sparse=False, ensure_2d=True, dtype=np.float64)
            self.return_label = True
            # sklearn ``metrics`` API needs attribute ``classes_``
            self.classes_ = np.unique(y) # Required for scikit-learn compatibility
            mask = (y == self.target)
            self.X_target = X[mask, :] if self.only_target else X
        else:
            X = check_array(X, ensure_2d=True, dtype=np.float64)
            self.X_target = X

        # Compute the kernel matrix
        self.G = pairwise_kernels(self.X_target, metric=self.metric, **self.kernel_params)
        self.n, self.m = self.X_target.shape
    
        # Compute the optimization matrix
        self.P = self._calculate_P(self.X_target)
        # Inirialize lower bounds
        self.lb = np.zeros((self.n, 1)) if self.lb is None else self.lb
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
        G = pairwise_kernels(X, Y, metric=metric, **self.kernel_params)

        # Compute adjustments for the kernel
        n, m = X.shape[0], Y.shape[0]
        G_kl = np.sum(G)
        Grow  = np.sum(G, axis=1, keepdims=True) # row-wise sum #G[i,:]  foreach i
        Gcol  = np.sum(G, axis=0, keepdims=True) # column-wise sum #G[:,j] foreach j
        
        # Broadcasting sums to match matrix dimensions
        Grow  = np.broadcast_to(Grow, shape=(n, m))
        Gcol = np.broadcast_to(Gcol,shape=(n,m))

        g = G - ( (Grow+Gcol)/n ) + (1/n**2)*G_kl
        #for i in tqdm(range(n),desc='BTB',leave=False):
        #    for j in range(m):
        #        g[i,j] = G[i,j] - (1/n)*( (np.sum(G[i,:])) + np.sum(G[:, j]) ) + (1/n**2)*G_kl
        
        return g # g is (n_sample_x, n_sample_y) matrix

    @ensure_fitted
    def __calculate_z__(self, sample):
        """
        Calcualte the z-value for a give sample.

        Parameters
        ----------
        sample : ndarray of shape (1, n_features)
            Input sample

        Returns
        -------
        z_value : float
            Computed z-value for the sample
        """
        # Compute kernel similarity between X_target and the sample
        Ky = self._adjust_kernel(self.X_target, sample)
        h  = Ky - (Ky.sum() + self.G.sum(axis=1, keepdims=True))/self.X_target.shape[0]
        h += self.G.sum()/(self.X_target.shape[0]**2)   
        
        # Prepare the optimization problem
        q = self.landa1*np.ones((self.n,1))+(-2*h)#??n or self.G.shape[0]?

        # Solve the quadratic programming problem
        x = solve_qp(2*self.P, q, lb=self.lb, solver=self.solver)

        # Compute the z-value
        z_value = self.landa1*x.sum() + self.landa2*np.linalg.norm(x)
        return z_value
    def _calculate_q(self, G: np.ndarray, Ky: np.ndarray) -> np.ndarray:
        """
        ONLY FOR THRESHOLDFINDER. NEED REFACTOR FOR ENETCONVEXHULL CLASS
        Calculate vector q for the quadratic problem.
        it is used for ThresholdFinder
        Parameters
        ----------
        G : ndarray of shape (n_samples, n_samples)
            Adjusted kernel matrix.
        Ky : ndarray of shape (n_samples, 1)
            Kernel similarity vector between X_target and a sample.

        Returns
        -------
        q : ndarray of shape (n_samples, 1)
            Vector q for quadratic programming.
        """
        n = G.shape[0]
        G_sum = G.sum()
        Ky_sum = Ky.sum()
        row_sum = G.sum(axis=1, keepdims=True)
        h = Ky - (Ky_sum + row_sum) / n + G_sum / (n ** 2)
        return self.landa1 * np.ones((n, 1)) - 2 * h
    @ensure_fitted
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
    @ensure_fitted
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
        # Validate input data
        X = check_array(X, ensure_2d=True, dtype=np.float64)

        # Compute decision scores
        scores = np.array([self.__calculate_z__(sample.reshape(1, -1)) for sample in X])
        return scores

    def set_kernel_params(self, **params):
        """
        Set additional parameter for kernel computations.

        Parameters
        ----------
        **params : dict
            Additional parameters for `pairwise_kernel`
        """
        self.kernel_params.update(params)

    def get_params(self, deep = True):
        """
        Get the parameters for this estimator.

        Parameters
        ----------
        deep : bool, optional(default=True)
            If True, return the parameter for this estimator and
            contained subobjects that are estimators.
        
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """

        return {
            'landa1'        : self.landa1,
            'target'        : self.target,
            'lb'            : self.lb,
            'solver'        : self.solver,
            'metric'        : self.metric,
            'only_target'   : self.only_target,
            'thr'           : self.thr,
            'kernel_params' : self.kernel_params,
            # Explicitly include kernel-specific parameters
            'degree'        : self.kernel_params.get('degree', None),
            'gamma'         : self.kernel_params.get('gamma', None),
            'coef0'         : self.kernel_params.get('coef0', None),
        }
    
    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        kernel_params = params.pop('kernel_params', {})
        for key in ['degree', 'gamma', 'coef0']:
            if key in params:
                kernel_params[key] = params.pop(key)
        
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter `{key}` for estimator `{self.__class__.__name__}`."
                                 "Check `get_params().keys()` for a list of valid parameters.")
        
        self.kernel_params.update(kernel_params)
        return self