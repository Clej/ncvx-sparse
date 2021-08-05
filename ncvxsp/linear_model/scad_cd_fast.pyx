# -*- coding: utf-8 -*-
"""
	Created on Wed Feb 24 16:52:53 2021

	@author: Clément Lejeune <clementlej@gmail.com>
"""
from libc.math cimport fabs, sqrt
cimport numpy as np
import numpy as np
import numpy.linalg as linalg

cimport cython
from cpython cimport bool
from cython cimport floating
import warnings
from sklearn.exceptions import ConvergenceWarning

from sklearn.utils._cython_blas cimport (_axpy, _dot, _asum, _ger, _gemv, _nrm2,
                                   _copy, _scal)
from sklearn.utils._cython_blas cimport RowMajor, ColMajor, Trans, NoTrans


from sklearn.utils._random cimport our_rand_r

ctypedef np.float64_t DOUBLE
ctypedef np.uint32_t UINT32_t

np.import_array()

# The following two functions are shamelessly copied from the tree code.

cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    RAND_R_MAX = 0x7FFFFFFF


cdef inline UINT32_t rand_int(UINT32_t end, UINT32_t* random_state) nogil:
    """Generate a random integer in [0; end)."""
    return our_rand_r(random_state) % end


cdef inline floating fmax(floating x, floating y) nogil:
    if x > y:
        return x
    return y


cdef inline floating fsign(floating f) nogil:
    if f == 0:
        return 0
    elif f > 0:
        return 1.0
    else:
        return -1.0


cdef floating max(int n, floating* a) nogil:
    """np.max(a)"""
    cdef int i
    cdef floating m = a[0]
    cdef floating d
    for i in range(1, n):
        d = a[i]
        if d > m:
            m = d
    return m
	
cdef floating abs_max(int n, floating* a) nogil:
    """np.max(np.abs(a))"""
    cdef int i
    cdef floating m = fabs(a[0])
    cdef floating d
    for i in range(1, n):
        d = fabs(a[i])
        if d > m:
            m = d
    return m


cdef floating diff_abs_max(int n, floating* a, floating* b) nogil:
    """np.max(np.abs(a - b))"""
    cdef int i
    cdef floating m = fabs(a[0] - b[0])
    cdef floating d
    for i in range(1, n):
        d = fabs(a[i] - b[i])
        if d > m:
            m = d
    return m


cdef floating prox_l1(floating x, floating alpha) nogil:
    """np.sign(x) * np.max(|x|-alpha, 0)"""
    cdef floating abs_x = fabs(x)
    if abs_x <= alpha:
        return 0.0
    else:
        return fsign(x) * (abs_x - alpha)
    
#========= SCAD regression ==========
cdef floating prox_scad(floating x, floating gam, floating alpha, floating beta, floating norm_X_ii) nogil:
    """
    Proximal operator of Smoothly Clipped Absolute Deviation (SCAD)
        x: input.
        gam: gamma parameter in scad, must be greater than 2.
        alpha: n_samples * l1 parameter, must be positive.
        beta: n_samples * 0.5*l2**2 parameter (ridge), must be positive.
        norm_X_ii: squared l2-norm of ii-th columns of design matrix.
                
        prox_scad(x, alpha, gam, beta) =
            if abs(x) <= alpha * (2 + beta):
                res = (sign(x) * max(abs(x) - gam, 0)) / (beta + norm_X_ii)
            elif alpha * (2 + beta) < abs(x) <= gam * alpha * (1 + beta):
                res = (gam-1)/(gam-2) * (sign(x) * max(abs(x) - alpha * gam / (gam-1), 0)) / (beta + norm_X_ii)
            else:
                res = x / (norm_X_ii + beta)
    """
    cdef floating alphaxgam
    cdef floating b1
    cdef floating b2
    cdef floating abs_x
    cdef floating sign_x
    b1 = 2*alpha + alpha #* beta => seems okay when beta!=0, passes check_estimators tests.
    b2 = alpha*gam #* (1.0 + beta)
    abs_x = fabs(x)
    # sign_x = fsign(x)
    if abs_x <= b1:
        return prox_l1(x, alpha) / (beta + norm_X_ii) # ridge-soft thresholding
    elif (b1 < abs_x) and (abs_x <= b2):
        #return sign_x * fmin(alphaxgam, fmax(alpha, (abs_x*(gam-1) - alphaxgam) / (gam-2 + beta * (gam-1))) )
        return (gam-1.0) / (gam-2.0) * prox_l1(x, alpha*gam/(gam-1.0)) / (beta + norm_X_ii) # ridge-soft/hard thresholding linear interpolation for abs_x in [b1, b2]
    elif abs_x > b2:
        return x / (beta + norm_X_ii) # ridge-hard thresholding
#--------------------------------
def scad_coordinate_descent(floating[::1] w,
                            floating alpha, floating beta,
                            floating gam,
                            floating[::1, :] X,
                            floating[::1] y,
                            int max_iter, floating tol,
                            object rng,
                            bint random=0, bint positive=0):
    """
        Cython version of the coordinate descent algorithm
        SCAD + ridge linear regression:
        minimize_w:
        (1/2) * l2_norm(y - X w, 2)^2 + alpha * SCAD(w, gam) + (beta/2) * l2_norm(w, 2)^2
    """

    if floating is float:
        dtype = np.float32
    else:
        dtype = np.float64

    # get the data information into easy vars
    cdef unsigned int n_samples = X.shape[0]
    cdef unsigned int n_features = X.shape[1]

    # compute norms of the columns of X
    cdef floating[::1] norm_cols_X = np.square(X).sum(axis=0)

    # initial value of the residuals
    cdef floating[::1] R = np.empty(n_samples, dtype=dtype)
    # cdef floating[::1] XtA = np.empty(n_features, dtype=dtype)

#    cdef floating gam
    cdef floating tmp
    cdef floating w_ii
    cdef floating d_w_max
    cdef floating w_max
    cdef floating d_w_ii
    cdef floating[::1] w_old = np.zeros(n_features, dtype=dtype)
    cdef floating[::1] diff_w = np.zeros(n_features, dtype=dtype)
    cdef floating w_old_norm
    cdef floating stationarity
    cdef floating gap = tol + 1.0
    cdef floating d_w_tol = tol
#    cdef floating dual_norm_XtA
#    cdef floating R_norm2
#    cdef floating w_norm2
#    cdef floating l1_norm
    cdef floating const
#    cdef floating A_norm2
    cdef unsigned int violations
    cdef unsigned int ii
    cdef unsigned int n_iter = 0
    cdef unsigned int f_iter
    cdef UINT32_t rand_r_state_seed = rng.randint(0, RAND_R_MAX)
    cdef UINT32_t* rand_r_state = &rand_r_state_seed

    if alpha == 0 and beta == 0:
        warnings.warn("SCAD Coordinate descent with no regularization may lead to "
                      "unexpected results and is discouraged. Instead use LinearRegression.")
        
    with nogil:
        
        # R = y - np.dot(X, w)
        _copy(n_samples, &y[0], 1, &R[0], 1)
        _gemv(ColMajor, NoTrans, n_samples, n_features, -1.0, &X[0, 0],
              n_samples, &w[0], 1, 1.0, &R[0], 1)

        # tol *= np.dot(y, y), to rescale tolerance with L2-norm of target samples
        tol = tol * _dot(n_samples, &y[0], 1, &y[0], 1)

        for n_iter in range(max_iter):
            w_max = 0.0
            d_w_max = 0.0
            _copy(n_features, &w[0], 1, &w_old[0], 1)
            
            for f_iter in range(n_features):  # Loop over coordinates
                if random:
                    ii = rand_int(n_features, rand_r_state)
                else:
                    ii = f_iter

                if norm_cols_X[ii] == 0.0 or w[ii] == 0.0: # Loop over active set
                    continue

                w_ii = w[ii]  # Store previous value

                # compute partial residuals (ie without feature ii)
                    # r = r + w_ii * X[:,ii] = y - y_fitted_except-on-ii
                _axpy(n_samples, w_ii, &X[0, ii], 1, &R[0], 1)

                # tmp = (X[:,ii]*R).sum(): sum of of partial residuals
                #  <=> univariate least-squares coef of fit (partial residuals vs. X[,ii])
                tmp = _dot(n_samples, &X[0, ii], 1, &R[0], 1)

                # proximal update of coef ii
                w[ii] = prox_scad(tmp, gam, alpha, beta, norm_cols_X[ii]) #/ norm_cols_X[ii]
                # w[ii] = (fsign(tmp) * fmax(fabs(tmp) - alpha, 0)
                #             / (norm_cols_X[ii] + beta))
                
                # Update residuals
                if w[ii] != 0.0:
                    # r = (r + w_ii * X[:,ii]) - w[ii] * X[:,ii] 
                    _axpy(n_samples, -w[ii], &X[0, ii], 1, &R[0], 1)

                # update the maximum absolute coefficient update
                d_w_ii = fabs(w[ii] - w_ii)
                d_w_max = fmax(d_w_max, d_w_ii)

                w_max = fmax(w_max, fabs(w[ii]))
                gap = 1.0 #TODO: manage case when gap is set to None as non-convexity of SCAD => primal and dual solution do not systematically coincide.
                
            # if d_w_max > tol and n_iter < max_iter - 1:
            #     continue
            # end active set update
            
            #----
            # SCAD path is not (unlike Elasticnet) piecewise-decreasing w.r.t alpha:
            # => Scanning violations out of active set in case some (inactive) coefficients get active (i.e. nonzero) after some iterations.
            violations = 0
            
            for f_iter in range(n_features):  # Loop over coordinates
                if random:
                    ii = rand_int(n_features, rand_r_state)
                else:
                    ii = f_iter

                if norm_cols_X[ii] == 0.0 or w[ii] != 0.0: # Loop over out inactive set
                    continue
            
                # partial residuals for feature ii
                # tmp = (X[:,ii]*R).sum()
                tmp = _dot(n_samples, &X[0, ii], 1, &R[0], 1)
                
                # update coef
                w[ii] = prox_scad(tmp, gam, alpha, beta, norm_cols_X[ii]) #/ norm_cols_X[ii]
                
                # update active set and residuals
                if w[ii] != 0.0:
                    # R -=  w[ii] * X[:,ii] 
                    _axpy(n_samples, -w[ii], &X[0, ii], 1, &R[0], 1)
                    w_ii = w[ii]
                    violations += 1
                    gap = violations *100
            ## End scan for violations
            
            if violations == 0:
                gap = 3.0
                
                if (w_max == 0.0 or 
                    d_w_max / w_max < d_w_tol or 
                    n_iter == max_iter - 1):
                    gap = 4.0
                    # Convergence check as relative l2-difference between current and previous iterates
                    _copy(n_features, &w[0], 1, &diff_w[0], 1)                  
                    _axpy(n_features, -1.0, &w_old[0], 1, &diff_w[0], 1)                    
                    stationarity = _nrm2(n_features, &diff_w[0], 1)
                    w_old_norm = _nrm2(n_features, &w_old[0], 1)
                    
                    if stationarity <= (d_w_tol * w_old_norm):
                        if w_old_norm != 0.0:
                            gap = stationarity / w_old_norm
                            break
                        else:
                            gap = stationarity
                            break
            
            # elif n_iter == max_iter:
            #     gap = 5.0
            #     break
                    
        else:
            # for/else, runs if for doesn't end with a `break`
            with gil: # TODO: precise stationarity value at divergence
                warnings.warn("SCAD regression: Objective did not converge.",
                              ConvergenceWarning)

    return w, gap, tol, n_iter + 1
#====================================
#
#==== group-l1 SCAD multi-task regression ======
cdef floating prox_gl1_scad(floating x, floating nn1, floating gam, floating l1_reg, floating l2_reg, floating norm_X_ii) nogil:
    
    # within-bounds of gl1-SCAD
    cdef floating b1 = 2*l1_reg - nn1
    cdef floating b2 = l1_reg*gam - nn1
    cdef floating abs_x = fabs(x)
    
    if abs_x <= b1:
        return prox_l1(x, l1_reg) / (l2_reg + norm_X_ii)
    
    elif (b1 < abs_x) and (abs_x <= b2):
        return (prox_l1(x, b2 / (gam-1.0)) / (1.0 - 1/(gam-1.0))) / (l2_reg + norm_X_ii)
    
    elif abs_x > b2:
        return x / (l2_reg + norm_X_ii)    
#---------------------
#==== group-l2 multi-task regression ======
cdef floating prox_l2(floating x, floating nn, floating l1_reg, floating l2_reg, floating norm_X_ii) nogil:
    
    return x * fmax(1 - l1_reg/nn, 0.0) / (norm_X_ii + l2_reg)
#---------------------
#==== group-l2 SCAD multi-task regression ======
cdef floating prox_gl2_scad(floating x, floating nn, floating gam, floating l1_reg, floating l2_reg, floating norm_X_ii) nogil:
    
    # within-bounds of gl2-SCAD
    cdef floating b1 = 2*l1_reg
    cdef floating b2 = l1_reg*gam
    
    if nn <= b1:
        return prox_l2(x, nn, l1_reg, l2_reg, norm_X_ii)
    
    elif b1 < nn and nn <= b2:
        return prox_l2(x, nn, l1_reg*gam / (gam-1.0), l2_reg, norm_X_ii) * (gam-1.0) / (gam-2.0)
    
    elif nn > b2:
        return x / (l2_reg + norm_X_ii)
#---------------------
cdef floating prox_gl_scad(bint task_sparsity_gl1, floating x, 
                           floating nn, floating gam, 
                           floating scad_reg, floating l2_reg, 
                           floating norm_X_ii) nogil:
    
    if task_sparsity_gl1 == 1:
        return prox_gl1_scad(x, nn, gam, scad_reg, l2_reg, norm_X_ii)
    
    elif task_sparsity_gl1 == 0:
        return prox_gl2_scad(x, nn, gam, scad_reg, l2_reg, norm_X_ii)
#---------------------

def scad_coordinate_descent_multi_task(
        floating[::1, :] W, floating scad_reg, floating l2_reg, 
        floating gam,
        np.ndarray[floating, ndim=2, mode='fortran'] X,  # TODO: use views with Cython 3.0
        np.ndarray[floating, ndim=2, mode='fortran'] Y,
        int max_iter, floating tol, object rng, bint random=0, bint task_sparsity_gl1=1):
    """
        Cython version of the coordinate descent algorithm
        for SCADnet multi-task regression

        We minimize w.r.t to W:

        0.5 * norm(Y - X W.T, 2)^2 + scad_reg \sum_i SCAD(norm(W_i, 1), gam) + 0.5 * l2_reg norm(W.T, 2)^2
        
        where W_i is the i-th feature vector over all the tasks.
    """

    if floating is float:
        dtype = np.float32
    else:
        dtype = np.float64

    # get the data information into easy vars
    cdef unsigned int n_samples = X.shape[0]
    cdef unsigned int n_features = X.shape[1]
    cdef unsigned int n_tasks = Y.shape[1]

    # # to store XtA
    # cdef floating[:, ::1] XtA = np.zeros((n_features, n_tasks), dtype=dtype)
    # cdef floating XtA_axis1norm
    # cdef floating dual_norm_XtA

    # initial value of the residuals
    cdef floating[::1, :] R = np.zeros((n_samples, n_tasks), dtype=dtype, order='F')

    cdef floating[::1] norm_cols_X = np.zeros(n_features, dtype=dtype)
    cdef floating[::1] tmp = np.zeros(n_tasks, dtype=dtype)
    cdef floating[::1] w_ii = np.zeros(n_tasks, dtype=dtype)
    cdef floating d_w_max
    cdef floating w_max
    cdef floating d_w_ii
    cdef floating nn # block-normalization in coef update
    cdef floating W_ii_abs_max
    cdef floating[::1, :] W_old = np.zeros((n_tasks, n_features), dtype=dtype, order='F')
    cdef floating[::1, :] diff_W = np.zeros((n_tasks, n_features), dtype=dtype, order='F')
    cdef floating w_old_norm
    cdef floating stationarity
    cdef floating gap = tol + 1.0 # we use this variable as output mode indicator
    cdef floating d_w_tol = tol
    # cdef floating R_norm
    # cdef floating w_norm
    # cdef floating ry_sum
    # cdef floating l21_norm
    cdef unsigned int violations
    cdef unsigned int ii
    cdef unsigned int jj
    cdef unsigned int n_iter = 0
    cdef unsigned int f_iter
    cdef UINT32_t rand_r_state_seed = rng.randint(0, RAND_R_MAX)
    cdef UINT32_t* rand_r_state = &rand_r_state_seed

    cdef floating* X_ptr = &X[0, 0]
    cdef floating* Y_ptr = &Y[0, 0]

    if scad_reg == 0:
        warnings.warn("Coordinate descent with l1_reg=0 may lead to unexpected"
            " results and is discouraged.")

    with nogil:
        # if task_sparsity_gl1 == 1:
        #     prox = prox_gl1_scad
        # else:
        #     prox = prox_gl2_scad
            
        # norm_cols_X = (np.asarray(X) ** 2).sum(axis=0)
        for ii in range(n_features):
            norm_cols_X[ii] = _nrm2(n_samples, X_ptr + ii * n_samples, 1) ** 2

        # R = Y - np.dot(X, W.T)
        _copy(n_samples * n_tasks, Y_ptr, 1, &R[0, 0], 1)
        for ii in range(n_features):
            for jj in range(n_tasks):
                if W[jj, ii] != 0:
                    _axpy(n_samples, -W[jj, ii], X_ptr + ii * n_samples, 1,
                          &R[0, jj], 1)

        # # tol = tol * linalg.norm(Y, ord='fro') ** 2, to rescale tolerance with L2-norm of target samples
        tol = tol * _nrm2(n_samples * n_tasks, Y_ptr, 1) ** 2

        for n_iter in range(max_iter):
            w_max = 0.0
            d_w_max = 0.0
            _copy(n_tasks * n_features, &W[0,0], 1, &W_old[0,0], 1)
            
            for f_iter in range(n_features):  # Loop over coordinates
                if random:
                    ii = rand_int(n_features, rand_r_state)
                else:
                    ii = f_iter

                if norm_cols_X[ii] == 0.0:
                    continue

                # w_ii = W[:, ii] # Store previous value
                _copy(n_tasks, &W[0, ii], 1, &w_ii[0], 1)

                # Partial residual update
                # Using Numpy:
                # R += np.dot(X[:, ii][:, None], w_ii[None, :]) # rank 1 update
                # Using Blas Level2:
                # _ger(RowMajor, n_samples, n_tasks, 1.0,
                #      &X[0, ii], 1,
                #      &w_ii[0], 1, &R[0, 0], n_tasks)
                # Using Blas Level1 and for loop to avoid slower threads
                # for such small vectors
                for jj in range(n_tasks):
                    if w_ii[jj] != 0:
                        _axpy(n_samples, w_ii[jj], X_ptr + ii * n_samples, 1,
                              &R[0, jj], 1)

                # Using numpy:
                # tmp = np.dot(X[:, ii][None, :], R).ravel()
                # Using BLAS Level 2:
                # _gemv(RowMajor, Trans, n_samples, n_tasks, 1.0, &R[0, 0],
                #       n_tasks, &X[0, ii], 1, 0.0, &tmp[0], 1)
                # Using BLAS Level 1 (faster for small vectors like here):
                for jj in range(n_tasks):
                    tmp[jj] = _dot(n_samples, X_ptr + ii * n_samples, 1,
                                    &R[0, jj], 1)
                
                if task_sparsity_gl1 == 1:
                    # nn = np.sum(abs(tmp))
                    nn = _asum(n_tasks, &tmp[0], 1)
                    
                elif task_sparsity_gl1 == 0:
                    # nn = sqrt(np.sum(tmp ** 2))
                    nn = _nrm2(n_tasks, &tmp[0], 1)
                
                # W[:, ii] = tmp * fmax(1. - l1_reg / nn, 0) / (norm_cols_X[ii] + l2_reg)
                # _copy(n_tasks, &tmp[0], 1, &W[0, ii], 1)
                # _scal(n_tasks, fmax(1. - l1_reg / nn, 0) / (norm_cols_X[ii] + l2_reg),
                #       &W[0, ii], 1)
                for jj in range(n_tasks):
                    # proximal update of coef ii for task jj
                    W[jj, ii] = prox_gl_scad(task_sparsity_gl1, tmp[jj], nn, gam, scad_reg, l2_reg, norm_cols_X[ii])
                        
                # Using numpy:
                # R -= np.dot(X[:, ii][:, None], W[:, ii][None, :])
                # Using BLAS Level 2:
                # Update residual : rank 1 update
                # _ger(RowMajor, n_samples, n_tasks, -1.0,
                #      &X[0, ii], 1, &W[0, ii], 1,
                #      &R[0, 0], n_tasks)
                # Using BLAS Level 1 (faster for small vectors like here):
                for jj in range(n_tasks):
                    if W[jj, ii] != 0:
                        _axpy(n_samples, -W[jj, ii], X_ptr + ii * n_samples, 1,
                              &R[0, jj], 1)

                # update the maximum absolute coefficient update
                d_w_ii = diff_abs_max(n_tasks, &W[0, ii], &w_ii[0])

                if d_w_ii > d_w_max:
                    d_w_max = d_w_ii

                W_ii_abs_max = abs_max(n_tasks, &W[0, ii])
                if W_ii_abs_max > w_max:
                    w_max = W_ii_abs_max
                 
                gap = 1.0 #TODO: manage case when gap is set to None as non-convexity of SCAD => primal and dual solution do not systematically coincide.
                
            # if d_w_max / w_max > d_w_tol and n_iter < max_iter - 1:
            #     continue
            
            # Scan for violations as inactive features may enter the active set,
            ## nonconvexity of SCAD => the regularization path is not piewewise-decreasing w.r.t scad_ratio.
            violations = 0
            
            for f_iter in range(n_features):  # Loop over coordinates
                if random:
                    ii = rand_int(n_features, rand_r_state)
                else:
                    ii = f_iter

                if norm_cols_X[ii] == 0.0:
                    continue
                          
                for jj in range(n_tasks):
                    if W[jj, ii] != 0.0: # Loop over out inactive set
                        continue
              
                    tmp[jj] = _dot(n_samples, X_ptr + ii * n_samples, 1,
                                    &R[0, jj], 1)
                      
                if task_sparsity_gl1 == 1:
                    # nn = np.sum(abs(tmp))
                    nn = _asum(n_tasks, &tmp[0], 1)
                    
                elif task_sparsity_gl1 == 0:
                    # nn = sqrt(np.sum(tmp ** 2))
                    nn = _nrm2(n_tasks, &tmp[0], 1)
                
                for jj in range(n_tasks):
                    if W[jj, ii] != 0.0:
                        continue
                    
                    # proximal update of coef ii for task jj
                    W[jj, ii] = prox_gl_scad(task_sparsity_gl1, tmp[jj], nn, gam, scad_reg, l2_reg, norm_cols_X[ii])
                
                    # if new active feature appears: update active set and residuals
                    if W[jj, ii] != 0:
                        _axpy(n_samples, -W[jj, ii], X_ptr + ii * n_samples, 1,
                              &R[0, jj], 1)
                        w_ii[jj] = W[jj, ii]
                        violations += 1
                        
                # update the maximum absolute coefficient update
                d_w_ii = diff_abs_max(n_tasks, &W[0, ii], &w_ii[0])

                if d_w_ii > d_w_max:
                    d_w_max = d_w_ii

                W_ii_abs_max = abs_max(n_tasks, &W[0, ii])
                if W_ii_abs_max > w_max:
                    w_max = W_ii_abs_max
            ## End scan for violations
            
            
            if violations == 0:
                gap = 3.0
            
                if (w_max == 0.0 or
                    d_w_max / w_max < d_w_tol or
                    n_iter == max_iter - 1):
                
                    # Convergence check as relative L2-difference between current and previous iterates
                    _copy(n_tasks * n_features, &W[0,0], 1, &diff_W[0,0], 1)                    
                    _axpy(n_tasks * n_features, -1.0, &W_old[0,0], 1, &diff_W[0,0], 1)                    
                    stationarity = _nrm2(n_tasks * n_features, &diff_W[0,0], 1)
                    w_old_norm = _nrm2(n_features * n_tasks, &W_old[0, 0], 1)
                    
                    if stationarity <= (d_w_tol * w_old_norm):
                        if w_old_norm != 0.0:
                            gap = stationarity / w_old_norm
                            break
                        else:
                            gap = stationarity
                            break
                    
            # elif n_iter == max_iter - 1:
            #     gap = 2.0
            #     break
            # else:
            #     continue
            
            # if (violations == 0):
            #     gap = 3.0
            #     # break        
                    
        else:
            gap = 2.0
            # for/else, runs if for doesn't end with a `break`
            with gil:
                warnings.warn("SCAD-l1-MTL objective did not converge. You might want to "
                              "increase the number of iterations or decrase tolerance. "
                              "tolerance: {}".format(tol),
                              ConvergenceWarning)

    return np.asarray(W), gap, tol, n_iter + 1
 