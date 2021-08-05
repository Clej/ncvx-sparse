# -*- coding: utf-8 -*-
"""
Created on Fri May 21 12:19:50 2021

@author: Clément Lejeune (clement.lejeune@irit.fr; clementlej@gmail.com)
"""

import sys
import warnings
import numbers
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy import sparse
from joblib import Parallel, effective_n_jobs

from sklearn.base import RegressorMixin, MultiOutputMixin
from sklearn.linear_model._base import LinearModel, _preprocess_data, _pre_fit
from sklearn.utils import check_array
from sklearn.utils.validation import check_random_state
from sklearn.model_selection import check_cv
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.fixes import _astype_copy_false, _joblib_parallel_args
from sklearn.utils.validation import check_is_fitted, _check_sample_weight
# from ..utils.validation import column_or_1d
# from ..utils.validation import _deprecate_positional_args
# from ..utils.fixes import delayed

from sklearn.linear_model._coordinate_descent import _alpha_grid, _set_order
from . import scad_cd_fast as scad_cd_fast

def scadnet_path(X, y, *, scad_ratio=0.5, gam = 3.7, eps=1e-3, n_alphas=100, alphas=None,
                 precompute='auto', Xy=None, copy_X=True, coef_init=None,
                 verbose=False, return_n_iter=False, positive=False,
                 check_input=True, **params):
    """
    Compute SCAD elastic net path with coordinate descent.

    The optimization function varies for mono and multi-outputs.

    For mono-output tasks it is::

        1 / (2 * n_samples) * ||y - Xw||^2_2
        + alpha * scad_ratio * SCAD(w, gam)
        + 0.5 * alpha * (1 - scad_ratio) * ||w||^2_2

    For multi-output tasks it is::

        (1 / (2 * n_samples)) * ||Y - XW||_Fro^2
        + alpha * SCAD_ratio * \\sum_i SCAD(||W_i||_1, gam)
        + 0.5 * alpha * (1 - scad_ratio) * ||W||_Fro^2

    Where::

        SCAD(u, gam) = 

    i.e. the L1 norm for small coefficients, the L0 pseudo-norm for large coefficients and a quadratic transition between both penalties.
    This is a continuous semi-concave (concave w.r.t |u|) version of the L0 pseudo-norm which is sparse and unbiased.

#    Read more in the :ref:`User Guide <elastic_net>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data to avoid
        unnecessary memory duplication. If ``y`` is mono-output then ``X``
        can be sparse.

    y : {array-like, sparse matrix} of shape (n_samples,) or \
        (n_samples, n_outputs)
        Target values.

    scad_ratio : float, default=0.5
        Number between 0 and 1 passed to SCAD net (scaling between
        SCAD and l2 penalties). ``scad_ratio=1`` corresponds to the SCAD regression without l2^2 regularization, a (biased) LASSO.

    eps : float, default=1e-3
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.

    n_alphas : int, default=100
        Number of alphas along the regularization path.

    alphas : ndarray, default=None
        List of alphas where to compute the models.
        If None alphas are set automatically.

    precompute : 'auto', bool or array-like of shape (n_features, n_features),\
                 default='auto'
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram
        matrix can also be passed as argument.

    Xy : array-like of shape (n_features,) or (n_features, n_outputs),\
         default=None
        Xy = np.dot(X.T, y) that can be precomputed. It is useful
        only when the Gram matrix is precomputed.

    copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.

    coef_init : ndarray of shape (n_features, ), default=None
        The initial values of the coefficients.

    verbose : bool or int, default=False
        Amount of verbosity.

    return_n_iter : bool, default=False
        Whether to return the number of iterations or not.

    positive : bool, default=False
        If set to True, forces coefficients to be positive.
        (Only allowed when ``y.ndim == 1``).

    check_input : bool, default=True
        If set to False, the input validation checks are skipped (including the
        Gram matrix when provided). It is assumed that they are handled
        by the caller.

    **params : kwargs
        Keyword arguments passed to the coordinate descent solver.

    Returns
    -------
    alphas : ndarray of shape (n_alphas,)
        The alphas along the path where models are computed.

    coefs : ndarray of shape (n_features, n_alphas) or \
            (n_outputs, n_features, n_alphas)
        Coefficients along the path.

    # dual_gaps : ndarray of shape (n_alphas,)
    #     The dual gaps at the end of the optimization for each alpha.

    n_iters : list of int
        The number of iterations taken by the coordinate descent optimizer to
        reach the specified tolerance for each alpha.
        (Is returned when ``return_n_iter`` is set to True).

    """
    # We expect X and y to be already Fortran ordered when bypassing
    # checks
    if check_input:
        X = check_array(X, accept_sparse='csc', dtype=[np.float64, np.float32],
                        order='F', copy=copy_X)
        y = check_array(y, accept_sparse='csc', dtype=X.dtype.type,
                        order='F', copy=False, ensure_2d=False)
        if Xy is not None:
            # Xy should be a 1d contiguous array or a 2D C ordered array
            Xy = check_array(Xy, dtype=X.dtype.type, order='C', copy=False,
                             ensure_2d=False)

    n_samples, n_features = X.shape

    # TODO: include task_sparsity ['group-l1', 'group-l2']
    multi_output = False
    if y.ndim != 1:
        multi_output = True
        _, n_outputs = y.shape
        task_sparsity_gl1 = 'group-l1'

    if multi_output and positive:
        raise ValueError('positive=True is not allowed for multi-output'
                         ' (y.ndim != 1)')

    # MultiTaskSCADnet does not support sparse matrices
    if not multi_output and sparse.isspmatrix(X):
        if 'X_offset' in params:
            # As sparse matrices are not actually centered we need this
            # to be passed to the CD solver.
            X_sparse_scaling = params['X_offset'] / params['X_scale']
            X_sparse_scaling = np.asarray(X_sparse_scaling, dtype=X.dtype)
        else:
            X_sparse_scaling = np.zeros(n_features, dtype=X.dtype)

    # X should be normalized and fit already if function is called
    # from ElasticNet.fit
    if check_input:
        X, y, X_offset, y_offset, X_scale, precompute, Xy = \
            _pre_fit(X, y, Xy, precompute, normalize=False,
                     fit_intercept=False, copy=False, check_input=check_input)
    if alphas is None:
        # No need to normalize of fit_intercept: it has been done
        # above
        alphas = _alpha_grid(X, y, Xy=Xy, l1_ratio=scad_ratio,
                             fit_intercept=False, eps=eps, n_alphas=n_alphas,
                             normalize=False, copy_X=False)
    else:
        alphas = np.sort(alphas)[::-1]  # make sure alphas are properly ordered

    n_alphas = len(alphas)
    tol = params.get('tol', 1e-4)
    max_iter = params.get('max_iter', 1000)
    dual_gaps = np.empty(n_alphas)
    n_iters = []

    rng = check_random_state(params.get('random_state', None))
    selection = params.get('selection', 'cyclic')
    if selection not in ['random', 'cyclic']:
        raise ValueError("selection should be either random or cyclic.")
    random = (selection == 'random')

    if not multi_output:
        coefs = np.empty((n_features, n_alphas), dtype=X.dtype)
    else:
        coefs = np.empty((n_outputs, n_features, n_alphas),
                         dtype=X.dtype)

    if coef_init is None:
        coef_ = np.zeros(coefs.shape[:-1], dtype=X.dtype, order='F')
    else:
        coef_ = np.asfortranarray(coef_init, dtype=X.dtype)

    for i, alpha in enumerate(alphas):
        # account for n_samples scaling in objectives between here and cd_fast
        scad_reg = alpha * scad_ratio * n_samples
        l2_reg = alpha * (1.0 - scad_ratio) * n_samples
        
        if not multi_output and sparse.isspmatrix(X):
            # model = cd_fast.sparse_enet_coordinate_descent(
            #     coef_, scad_reg, l2_reg, X.data, X.indices,
            #     X.indptr, y, X_sparse_scaling,
            #     max_iter, tol, rng, random, positive)
            
            raise ValueError("SCADnet cannot deal sparse matrices and may do it in future version. "
                             "Instead, use Lasso or ElasticNet.")
        elif multi_output:
            # if gam is None:
            #     model = cd_fast.enet_coordinate_descent_multi_task(
            #         coef_, scad_reg, l2_reg, X, y, max_iter, tol, rng, random)
            # else:
            model = scad_cd_fast.scad_coordinate_descent_multi_task(
                coef_, scad_reg, l2_reg, gam,
                X, y,
                max_iter, tol, rng, random, task_sparsity_gl1)
        elif isinstance(precompute, np.ndarray):
            # We expect precompute to be already Fortran ordered when bypassing
            # checks
            if check_input:
                precompute = check_array(precompute, dtype=X.dtype.type,
                                         order='C')
            # model = cd_fast.enet_coordinate_descent_gram(
            #     coef_, scad_reg, l2_reg, precompute, Xy, y, max_iter,
            #     tol, rng, random, positive)
            
            raise ValueError("SCADnet cannot deal Gram matrix as input and may do it in future version. "
                             "Instead, use Lasso or ElasticNet.")
        elif precompute is False:
                model = scad_cd_fast.scad_coordinate_descent(
                    coef_, scad_reg, l2_reg, gam, X, y, max_iter, tol, rng, random,
                    positive)
        else:
            raise ValueError("Precompute should be one of True, False, "
                             "'auto' or array-like. Got %r" % precompute)
        coef_, dual_gap_, eps_, n_iter_ = model
        coefs[..., i] = coef_
        # we correct the scale of the returned dual gap, as the objective
        # in cd_fast is n_samples * the objective in this docstring.
        dual_gaps[i] = dual_gap_ / n_samples
        n_iters.append(n_iter_)

        if verbose:
            if verbose > 2:
                print(model)
            elif verbose > 1:
                print('Path: %03i out of %03i' % (i, n_alphas))
            else:
                sys.stderr.write('.')

    if return_n_iter:
        return alphas, coefs, dual_gaps, n_iters
    return alphas, coefs, dual_gaps

###############################################################################
# SCAD + L2 linear regression


class SCADnet(MultiOutputMixin, RegressorMixin, LinearModel):
    """Linear regression combined with SCAD and ridge priors as regularizers.

    The optimization objective for SCADnet regression is::


        1 / (2 * n_samples) * ||y - Xw||^2_2
        + alpha * scad_ratio * SCAD(w, gam)
        + 0.5 * alpha * (1 - scad_ratio) * ||w||^2_2
        
    Where::

        SCAD(w, gam) = 

    i.e. the L1 norm for small coefficients (in absolute value), the L0 norm for large coefficients and a quadratic transition between both penalties.
    This is a continuous semi-concave (concave w.r.t |u|) relaxation of the L0 pseudo-norm which is sparse and unbiased.
    """
    path = staticmethod(scadnet_path)

    # @_deprecate_positional_args
    def __init__(self, alpha=1.0, *, scad_ratio=0.5,
                 gam = 3.7,
                 fit_intercept=True,
                 normalize=False, precompute=False, max_iter=1000,
                 copy_X=True, tol=1e-4, warm_start=False, positive=False,
                 random_state=None, selection='cyclic'):
        self.alpha = alpha
        self.scad_ratio = scad_ratio
        self.gam = gam
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.precompute = precompute
        self.max_iter = max_iter
        self.copy_X = copy_X
        self.tol = tol
        self.warm_start = warm_start
        self.positive = positive # positive=True, currently not available for SCAD
        self.random_state = random_state
        self.selection = selection

    def fit(self, X, y, sample_weight=None, check_input=True):
        """Fit model with coordinate descent.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of (n_samples, n_features)
            Data.

        y : {ndarray, sparse matrix} of shape (n_samples,) or \
            (n_samples, n_targets)
            Target. Will be cast to X's dtype if necessary.

        sample_weight : float or array-like of shape (n_samples,), default=None
            Sample weight. Internally, the `sample_weight` vector will be
            rescaled to sum to `n_samples`.

            .. versionadded:: 0.23

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Notes
        -----

        Coordinate descent is an algorithm that considers each column of
        data at a time hence it will automatically convert the X input
        as a Fortran-contiguous numpy array if necessary.

        To avoid memory re-allocation it is advised to allocate the
        initial data in memory directly using that format.
        """

        if self.alpha == 0:
            warnings.warn("With alpha=0, this algorithm does not converge "
                          "well. You are advised to use the LinearRegression "
                          "estimator", stacklevel=2)
        
        if isinstance(self.precompute, str):
            raise ValueError('precompute should be one of True, False or'
                             ' array-like. Got %r' % self.precompute)

        if (not isinstance(self.scad_ratio, numbers.Number) or
                self.scad_ratio < 0 or self.scad_ratio > 1):
            raise ValueError("scad_ratio must be between 0 and 1; "
                             f"got scad_ratio={self.scad_ratio}")
            
        # if self.scad_ratio < 1.0:
        #     raise ValueError("Set scad_ratio < 1 when some columns in matrix 'X' are highly correlated, "
        #                      "otherwise SCAD coordinate descent may not converge well. "
        #                      "Note that for scad_ratio < 1, coefficients are biased (less than with ElasticNet).")
        
        if (not isinstance(self.gam, numbers.Number) or
                self.gam < 2):
            raise ValueError("gam parameter must be greater than 2; "
                             f"got gam={self.gam}. "
                             "It is recommended to use default value 3.7.")

        # Remember if X is copied
        X_copied = False
        # We expect X and y to be float64 or float32 Fortran ordered arrays
        # when bypassing checks
        if check_input:
            X_copied = self.copy_X and self.fit_intercept
            X, y = self._validate_data(X, y, accept_sparse='csc',
                                       order='F',
                                       dtype=[np.float64, np.float32],
                                       copy=X_copied, multi_output=True,
                                       y_numeric=True)
            y = check_array(y, order='F', copy=False, dtype=X.dtype.type,
                            ensure_2d=False)

        n_samples, n_features = X.shape
        alpha = self.alpha

        if isinstance(sample_weight, numbers.Number):
            sample_weight = None
        if sample_weight is not None:
            if check_input:
                if sparse.issparse(X):
                    raise ValueError("Sample weights do not (yet) support "
                                     "sparse matrices.")
                sample_weight = _check_sample_weight(sample_weight, X,
                                                     dtype=X.dtype)
            # simplify things by rescaling sw to sum up to n_samples
            # => np.average(x, weights=sw) = np.mean(sw * x)
            sample_weight = sample_weight * (n_samples / np.sum(sample_weight))
            # Objective function is:
            # 1/2 * np.average(squared error, weights=sw) + alpha * penalty
            # but coordinate descent minimizes:
            # 1/2 * sum(squared error) + alpha * penalty
            # enet_path therefore sets alpha = n_samples * alpha
            # With sw, enet_path should set alpha = sum(sw) * alpha
            # Therefore, we rescale alpha = sum(sw) / n_samples * alpha
            # Note: As we rescaled sample_weights to sum up to n_samples,
            #       we don't need this
            # alpha *= np.sum(sample_weight) / n_samples

        # Ensure copying happens only once, don't do it again if done above.
        # X and y will be rescaled if sample_weight is not None, order='F'
        # ensures that the returned X and y are still F-contiguous.
        should_copy = self.copy_X and not X_copied
        X, y, X_offset, y_offset, X_scale, precompute, Xy = \
            _pre_fit(X, y, None, self.precompute, self.normalize,
                     self.fit_intercept, copy=should_copy,
                     check_input=check_input, sample_weight=sample_weight)
        # coordinate descent needs F-ordered arrays and _pre_fit might have
        # called _rescale_data
        if check_input or sample_weight is not None:
            X, y = _set_order(X, y, order='F')
        if y.ndim == 1:
            y = y[:, np.newaxis]
            
        if Xy is not None and Xy.ndim == 1:
            Xy = Xy[:, np.newaxis]

        n_targets = y.shape[1]

        if self.selection not in ['cyclic', 'random']:
            raise ValueError("selection should be either random or cyclic.")

        if not self.warm_start or not hasattr(self, "coef_"):
            coef_ = np.zeros((n_targets, n_features), dtype=X.dtype,
                             order='F')
        else:
            coef_ = self.coef_
            if coef_.ndim == 1:
                coef_ = coef_[np.newaxis, :]

        dual_gaps_ = np.zeros(n_targets, dtype=X.dtype)
        self.n_iter_ = []

        # fits n_targets single-output linear regressions independently
        for k in range(n_targets):
            if Xy is not None:
                this_Xy = Xy[:, k]
            else:
                this_Xy = None
            _, this_coef, this_dual_gap, this_iter = \
                self.path(X, y[:, k],
                          scad_ratio=self.scad_ratio, eps=None,
                          gam=self.gam,
                          n_alphas=None, alphas=[alpha],
                          precompute=precompute, Xy=this_Xy,
                          fit_intercept=False, normalize=False, copy_X=True,
                          verbose=False, tol=self.tol, positive=self.positive,
                          X_offset=X_offset, X_scale=X_scale,
                          return_n_iter=True, coef_init=coef_[k],
                          max_iter=self.max_iter,
                          random_state=self.random_state,
                          selection=self.selection,
                          check_input=False)
            coef_[k] = this_coef[:, 0]
            dual_gaps_[k] = this_dual_gap[0]
            self.n_iter_.append(this_iter[0])

        if n_targets == 1:
            self.n_iter_ = self.n_iter_[0]
            self.coef_ = coef_[0]
            self.dual_gap_ = dual_gaps_[0]
        else:
            self.coef_ = coef_
            self.dual_gap_ = dual_gaps_

        self._set_intercept(X_offset, y_offset, X_scale)

        # workaround since _set_intercept will cast self.coef_ into X.dtype
        self.coef_ = np.asarray(self.coef_, dtype=X.dtype)

        # return self for chaining fit and predict calls
        return self

    @property
    def sparse_coef_(self):
        """Sparse representation of the fitted `coef_`."""
        return sparse.csr_matrix(self.coef_)

    def _decision_function(self, X):
        """Decision function of the linear model.

        Parameters
        ----------
        X : numpy array or scipy.sparse matrix of shape (n_samples, n_features)

        Returns
        -------
        T : ndarray of shape (n_samples,)
            The predicted decision function.
        """
        check_is_fitted(self)
        if sparse.isspmatrix(X):
            return safe_sparse_dot(X, self.coef_.T,
                                   dense_output=True) + self.intercept_
        else:
            return super()._decision_function(X)
        
class MultiTaskSCADnet(SCADnet):

    """Multi-task SCADNet model trained with SCAD/L2 mixed-norm as
    regularizer.

    The optimization objective for MultiTaskSCADnet is::

        (1 / (2 * n_samples)) * ||Y - XW||_Fro^2
        + alpha * SCAD_ratio * \\sum_i SCAD(||W_i||_1, gam)
        + 0.5 * alpha * (1 - scad_ratio) * ||W||_Fro^2

    Where::

        SCAD(||W_i||_1, gam) = 

    i.e. the sum of SCAD of the l1 norm of each row.

    Read more in the :ref:`User Guide <multi_task_elastic_net>`.

    Parameters
    ----------
    alpha : float, default=1.0
        Constant that multiplies the SCAD/L2 term. Defaults to 1.0.

     : float, default=0.5
        The MultiSCADnet mixing parameter, with 0 < scad_ratio <= 1.
        For scad_ratio = 1 the penalty is an SCAD/L2 penalty. For scad_ratio = 0 it
        is an L2 penalty.
        For ``0 < scad_ratio < 1``, the penalty is a combination of SCAD/L1 and L2.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    normalize : bool, default=False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`~sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

    copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.

    max_iter : int, default=1000
        The maximum number of iterations.

    tol : float, default=1e-4
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        See :term:`the Glossary <warm_start>`.

    random_state : int, RandomState instance, default=None
        The seed of the pseudo random number generator that selects a random
        feature to update. Used when ``selection`` == 'random'.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    selection : {'cyclic', 'random'}, default='cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.

    Attributes
    ----------
    intercept_ : ndarray of shape (n_tasks,)
        Independent term in decision function.

    coef_ : ndarray of shape (n_tasks, n_features)
        Parameter vector (W in the cost function formula). If a 1D y is
        passed in at fit (non multi-task usage), ``coef_`` is then a 1D array.
        Note that ``coef_`` stores the transpose of ``W``, ``W.T``.

    n_iter_ : int
        Number of iterations run by the coordinate descent solver to reach
        the specified tolerance.

    dual_gap_ : float
        The dual gaps at the end of the optimization.

    eps_ : float
        The tolerance scaled scaled by the variance of the target `y`.

    sparse_coef_ : sparse matrix of shape (n_features,) or \
            (n_tasks, n_features)
        Sparse representation of the `coef_`.


    Notes
    -----
    The algorithm used to fit the model is coordinate descent.

    To avoid unnecessary memory duplication the X and y arguments of the fit
    method should be directly passed as Fortran-contiguous numpy arrays.
    """
    # @_deprecate_positional_args
    def __init__(self, alpha=1.0, *, scad_ratio=0.5, gam=3.7, task_sparsity = 'group-l1', fit_intercept=True,
                 normalize=False, copy_X=True, max_iter=1000, tol=1e-4,
                 warm_start=False, random_state=None, selection='cyclic'):
        self.scad_ratio = scad_ratio
        self.gam = gam
        self.alpha = alpha
        self.task_sparsity = task_sparsity
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.max_iter = max_iter
        self.copy_X = copy_X
        self.tol = tol
        self.warm_start = warm_start
        self.random_state = random_state
        self.selection = selection

    def fit(self, X, y):
        """Fit SCADNet model with coordinate descent

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data.
        y : ndarray of shape (n_samples, n_tasks)
            Target. Will be cast to X's dtype if necessary.

        Notes
        -----

        Coordinate descent is an algorithm that considers each column of
        data at a time hence it will automatically convert the X input
        as a Fortran-contiguous numpy array if necessary.

        To avoid memory re-allocation it is advised to allocate the
        initial data in memory directly using that format.
        """
        if (self.task_sparsity not in ['group-l1', 'group-l2']) or (not isinstance(self.task_sparsity, str)):
            raise ValueError("task_sparsity should be a string, either 'group-l1' or 'group-l2'; "
                             f"got gam={self.task_sparsity}. ")
        
        task_sparsity_gl1 = (self.task_sparsity == 'group-l1')
        
        if (not isinstance(self.gam, numbers.Number) or
                self.gam < 2):
            raise ValueError("gam parameter must be greater than 2; "
                             f"got gam={self.gam}. ")
            
        # Need to validate separately here.
        # We can't pass multi_ouput=True because that would allow y to be csr.
        check_X_params = dict(dtype=[np.float64, np.float32], order='F',
                              copy=self.copy_X and self.fit_intercept)
        check_y_params = dict(ensure_2d=False, order='F')
        X, y = self._validate_data(X, y, validate_separately=(check_X_params,
                                                              check_y_params))
        y = y.astype(X.dtype)

        if hasattr(self, 'scad_ratio'):
            model_str = 'ElasticNet'
        else:
            model_str = 'Lasso'
        if y.ndim == 1:
            raise ValueError("For mono-task outputs, use %s" % model_str)

        n_samples, n_features = X.shape
        _, n_tasks = y.shape

        if n_samples != y.shape[0]:
            raise ValueError("X and y have inconsistent dimensions (%d != %d)"
                             % (n_samples, y.shape[0]))

        X, y, X_offset, y_offset, X_scale = _preprocess_data(
            X, y, self.fit_intercept, self.normalize, copy=False)

        if not self.warm_start or not hasattr(self, "coef_"):
            self.coef_ = np.zeros((n_tasks, n_features), dtype=X.dtype.type,
                                  order='F')

        scad_reg = self.alpha * self.scad_ratio * n_samples
        l2_reg = self.alpha * (1.0 - self.scad_ratio) * n_samples

        self.coef_ = np.asfortranarray(self.coef_)  # coef contiguous in memory

        if self.selection not in ['random', 'cyclic']:
            raise ValueError("selection should be either random or cyclic.")
        random = (self.selection == 'random')
        
        self.coef_, self.dual_gap_, self.eps_, self.n_iter_ = \
            scad_cd_fast.scad_coordinate_descent_multi_task(
                self.coef_, scad_reg, l2_reg, self.gam,
                X, y,
                self.max_iter, self.tol, check_random_state(self.random_state), random, task_sparsity_gl1)
            
        # account for different objective scaling here and in cd_fast
        self.dual_gap_ /= n_samples

        self._set_intercept(X_offset, y_offset, X_scale)

        # return self for chaining fit and predict calls
        return self

    def _more_tags(self):
        return {'multioutput_only': True}
