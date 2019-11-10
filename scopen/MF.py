""" Bounded Non-negative Matrix Factorization
"""
from math import sqrt
import warnings
import numbers
import numpy as np
import scipy.sparse as sp

from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import randomized_svd, safe_sparse_dot, squared_norm
from sklearn.exceptions import ConvergenceWarning

from cdnmf_fast import _update_cdnmf_fast

EPSILON = np.finfo(np.float16).eps

INTEGER_TYPES = (numbers.Integral, np.integer)


def norm(x):
    """Dot product-based Euclidean norm implementation

    See: http://fseoane.net/blog/2011/computing-the-vector-norm/

    Parameters
    ----------
    x : array-like
        Vector for which to compute the norm
    """
    return sqrt(squared_norm(x))


def _check_non_negative(X, whom):
    """
    Check if there is any negative value in an array.

    Parameters
    ----------
    X : array-like or sparse matrix
        Input data.

    whom : string
        Who passed X to this function.
    """
    X = X.data if sp.issparse(X) else X
    if (X < 0).any():
        raise ValueError("Negative values in data passed to %s" % whom)


def _safe_squared_norm(X):
    """Squared Euclidean or Frobenius norm of X, safe for numpy masked arrays.
    """
    X = _safe_ravel(X)
    if isinstance(X, np.ma.masked_array):
        return float(np.ma.dot(X, X))
    else:
        return np.dot(X, X)


def _safe_ravel(X):
    """Guarantee that we preserve masked array in ravel.
    """
    if isinstance(X, np.ma.masked_array):
        return np.ma.ravel(X)
    else:
        return np.ravel(X)


def _safe_mean(X):
    """Compute the arithmetic mean of an array, safe for sparse and NaN values
    """
    if sp.issparse(X):
        return X.mean()
    else:
        return np.mean(X)


def _initialize_nmf(X, n_components, init=None, eps=1e-6,
                    random_state=None):
    """Algorithms for NMF initialization.

    Computes an initial guess for the non-negative
    rank k matrix approximation for X: X = WH

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data matrix to be decomposed.

    n_components : integer
        The number of components desired in the approximation.

    init :  None | 'random' | 'nndsvd' | 'nndsvda' | 'nndsvdar'
        Method used to initialize the procedure.
        Default: 'nndsvd' if n_components < n_features, otherwise 'random'.
        Valid options:

        - 'random': non-negative random matrices, scaled with:
            sqrt(X.mean() / n_components)

        - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
            initialization (better for sparseness)

        - 'nndsvda': NNDSVD with zeros filled with the average of X
            (better when sparsity is not desired)

        - 'nndsvdar': NNDSVD with zeros filled with small random values
            (generally faster, less accurate alternative to NNDSVDa
            for when sparsity is not desired)

        - 'custom': use custom matrices W and H

    eps : float
        Truncate all values less then this in output to zero.

    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Used when ``random`` == 'nndsvdar' or 'random'.

    Returns
    -------
    W : array-like, shape (n_samples, n_components)
        Initial guesses for solving X ~= WH

    H : array-like, shape (n_components, n_features)
        Initial guesses for solving X ~= WH

    References
    ----------
    C. Boutsidis, E. Gallopoulos: SVD based initialization: A head start for
    nonnegative matrix factorization - Pattern Recognition, 2008
    http://tinyurl.com/nndsvd
    """
    _check_non_negative(X, "NMF initialization")
    n_samples, n_features = X.shape

    if init is None:
        if n_components < n_features:
            init = 'nndsvd'
        else:
            init = 'random'

    # Random initialization
    if init == 'random':
        X_mean = _safe_mean(X)
        avg = np.sqrt(X_mean / n_components)
        rng = check_random_state(random_state)
        H = avg * rng.randn(n_components, n_features)
        W = avg * rng.randn(n_samples, n_components)
        # we do not write np.abs(H, out=H) to stay compatible with
        # numpy 1.5 and earlier where the 'out' keyword is not
        # supported as a kwarg on ufuncs
        np.abs(H, H)
        np.abs(W, W)
        return W, H

    if not sp.issparse(X) and np.any(np.isnan(X)):
        raise ValueError("NMF initializations with NNDSVD are not available "
                         "with missing values (np.nan).")

    # NNDSVD initialization
    U, S, V = randomized_svd(X, n_components, random_state=random_state)
    W, H = np.zeros(U.shape), np.zeros(V.shape)

    # The leading singular triplet is non-negative
    # so it can be used as is for initialization.
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

    for j in range(1, n_components):
        x, y = U[:, j], V[j, :]

        # extract positive and negative parts of column vectors
        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

        # and their norms
        x_p_nrm, y_p_nrm = norm(x_p), norm(y_p)
        x_n_nrm, y_n_nrm = norm(x_n), norm(y_n)

        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

        # choose update
        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n

        lbd = np.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v

    W[W < eps] = 0
    H[H < eps] = 0

    if init == "nndsvd":
        pass
    elif init == "nndsvda":
        avg = X.mean()
        W[W == 0] = avg
        H[H == 0] = avg
    elif init == "nndsvdar":
        rng = check_random_state(random_state)
        avg = X.mean()
        W[W == 0] = abs(avg * rng.randn(len(W[W == 0])) / 100)
        H[H == 0] = abs(avg * rng.randn(len(H[H == 0])) / 100)
    else:
        raise ValueError(
            'Invalid init parameter: got %r instead of one of %r' %
            (init, (None, 'random', 'nndsvd', 'nndsvda', 'nndsvdar')))

    return W, H


def _update_coordinate_descent(X, W, Ht, l1_reg, l2_reg, shuffle,
                               random_state):
    """Helper function for _fit_coordinate_descent

    Update W to minimize the objective function, iterating once over all
    coordinates. By symmetry, to update H, one can call
    _update_coordinate_descent(X.T, Ht, W, ...)

    """
    n_components = Ht.shape[1]

    HHt = np.dot(Ht.T, Ht)
    XHt = safe_sparse_dot(X, Ht)

    # L1 regularization corresponds to decrease of each element of XHt
    if l1_reg != 0.:
        XHt -= l1_reg

    # L2 regularization corresponds to increase of the diagonal of HHt
    if l2_reg != 0.:
        # adds l2_reg only on the diagonal
        HHt.flat[::n_components + 1] += l2_reg

    if shuffle:
        permutation = random_state.permutation(n_components)
    else:
        permutation = np.arange(n_components)

    # The following seems to be required on 64-bit Windows w/ Python 3.5.
    permutation = np.asarray(permutation, dtype=np.intp)

    return _update_cdnmf_fast(W, HHt, XHt, permutation)


def _fit_coordinate_descent(X, W, H, tol=1e-8, max_iter=200, l2_reg_W=0, l2_reg_H=0,
                            verbose=0, shuffle=False, random_state=None):
    """Compute Non-negative Matrix Factorization (NMF) with Coordinate Descent

    The objective function is minimized with an alternating minimization of W
    and H. Each minimization is done with a cyclic (up to a permutation of the
    features) Coordinate Descent.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Constant matrix.

    W : array-like, shape (n_samples, n_components)
        Initial guess for the solution.

    H : array-like, shape (n_components, n_features)
        Initial guess for the solution.

    tol : float, default: 1e-4
        Tolerance of the stopping condition.

    max_iter : integer, default: 200
        Maximum number of iterations before timing out.

    l2_reg_W : double, default: 0.
        L2 regularization parameter for W.

    l2_reg_H : double, default: 0.
        L2 regularization parameter for H.

    verbose : integer, default: 0
        The verbosity level.

    shuffle : boolean, default: False
        If true, randomize the order of coordinates in the CD solver.

    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    W : array-like, shape (n_samples, n_components)
        Solution to the non-negative least squares problem.

    H : array-like, shape (n_components, n_features)
        Solution to the non-negative least squares problem.

    n_iter : int
        The number of iterations done by the algorithm.

    References
    ----------
    Cichocki, Andrzej, and Phan, Anh-Huy. "Fast local algorithms for
    large scale nonnegative matrix and tensor factorizations."
    IEICE transactions on fundamentals of electronics, communications and
    computer sciences 92.3: 708-721, 2009.
    """
    # so W and Ht are both in C order in memory
    Ht = check_array(H.T, order='C')
    X = check_array(X, accept_sparse='csr')

    rng = check_random_state(random_state)

    n_iter = 0
    violation_init = 0

    for n_iter in range(max_iter):
        violation = 0.

        # Update W and H
        violation += _update_coordinate_descent(X, W, Ht, 0, l2_reg_W, shuffle, rng)
        violation += _update_coordinate_descent(X.T, Ht, W, 0, l2_reg_H, shuffle, rng)

        if n_iter == 0:
            violation_init = violation

        if violation_init == 0:
            break

        if verbose:
            print("violation:", violation / violation_init)

        if violation / violation_init <= tol:
            if verbose:
                print("Converged at iteration", n_iter + 1)
            break

    return W, Ht.T, n_iter


def bounded_non_negative_factorization(X, n_components=None, init='random',
                                       tol=1e-8, max_iter=200, alpha=0., random_state=None,
                                       verbose=0, shuffle=False):
    r"""Compute Bounded Non-negative Matrix Factorization (BNMF)

    Find two non-negative matrices (W, H) whose product approximates the non-
    negative matrix X. This factorization can be used for example for
    dimensionality reduction, source separation or topic extraction.

    The objective function is::

        0.5 * ||X - WH||_Fro^2
        + alpha * l1_ratio * ||vec(W)||_1
        + alpha * l1_ratio * ||vec(H)||_1
        + 0.5 * alpha * (1 - l1_ratio) * ||W||_Fro^2
        + 0.5 * alpha * (1 - l1_ratio) * ||H||_Fro^2

    Where::

        ||A||_Fro^2 = \sum_{i,j} A_{ij}^2 (Frobenius norm)
        ||vec(A)||_1 = \sum_{i,j} abs(A_{ij}) (Elementwise L1 norm)
        0 <= W, H <= 1.0 / sqrt(n_components)

    For multiplicative-update ('mu') solver, the Frobenius norm
    (0.5 * ||X - WH||_Fro^2) can be changed into another beta-divergence loss,
    by changing the beta_loss parameter.

    The objective function is minimized with an alternating minimization of W
    and H. If H is given and update_H=False, it solves for W only.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Constant matrix.

    W : array-like, shape (n_samples, n_components)
        If init='custom', it is used as initial guess for the solution.

    H : array-like, shape (n_components, n_features)
        If init='custom', it is used as initial guess for the solution.
        If update_H=False, it is used as a constant, to solve for W only.

    n_components : integer
        Number of components, if n_components is not set all features
        are kept.

    init :  None | 'random' | 'nndsvd' | 'nndsvda' | 'nndsvdar' | 'custom'
        Method used to initialize the procedure.
        Default: 'nndsvd' if n_components < n_features, otherwise random.
        Valid options:

        - 'random': non-negative random matrices, scaled with:
            sqrt(X.mean() / n_components)

        - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
            initialization (better for sparseness)

        - 'nndsvda': NNDSVD with zeros filled with the average of X
            (better when sparsity is not desired)

        - 'nndsvdar': NNDSVD with zeros filled with small random values
            (generally faster, less accurate alternative to NNDSVDa
            for when sparsity is not desired)

        - 'custom': use custom matrices W and H

    update_H : boolean, default: True
        Set to True, both W and H will be estimated from initial guesses.
        Set to False, only W will be estimated.

    tol : float, default: 1e-4
        Tolerance of the stopping condition.

    max_iter : integer, default: 200
        Maximum number of iterations before timing out.

    alpha : double, default: 0.
        Constant that multiplies the regularization terms.

    l1_ratio : double, default: 0.
        The regularization mixing parameter, with 0 <= l1_ratio <= 1.
        For l1_ratio = 0 the penalty is an elementwise L2 penalty
        (aka Frobenius Norm).
        For l1_ratio = 1 it is an elementwise L1 penalty.
        For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.

    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : integer, default: 0
        The verbosity level.

    shuffle : boolean, default: False
        If true, randomize the order of coordinates in the CD solver.

    Returns
    -------
    W : array-like, shape (n_samples, n_components)
        Solution to the non-negative least squares problem.

    H : array-like, shape (n_components, n_features)
        Solution to the non-negative least squares problem.

    n_iter : int
        Actual number of iterations.

    References
    ----------
    Cichocki, Andrzej, and P. H. A. N. Anh-Huy. "Fast local algorithms for
    large scale nonnegative matrix and tensor factorizations."
    IEICE transactions on fundamentals of electronics, communications and
    computer sciences 92.3: 708-721, 2009.

    Fevotte, C., & Idier, J. (2011). Algorithms for nonnegative matrix
    factorization with the beta-divergence. Neural Computation, 23(9).
    """
    X = check_array(X, accept_sparse=('csr', 'csc'), dtype=float,
                    force_all_finite=False)
    _check_non_negative(X, "NMF (input X)")

    if sp.issparse(X) and np.any(np.isnan(X.data)):
        raise ValueError("X contains NaN values, and NMF with missing "
                         "values is not implemented for sparse matrices.")

    n_samples, n_features = X.shape
    if n_components is None:
        n_components = n_features

    if not isinstance(n_components, INTEGER_TYPES) or n_components <= 0:
        raise ValueError("Number of components must be a positive integer;"
                         " got (n_components=%r)" % n_components)
    if not isinstance(max_iter, INTEGER_TYPES) or max_iter < 0:
        raise ValueError("Maximum number of iterations must be a positive "
                         "integer; got (max_iter=%r)" % max_iter)
    if not isinstance(tol, numbers.Number) or tol < 0:
        raise ValueError("Tolerance for stopping criteria must be "
                         "positive; got (tol=%r)" % tol)

    # check W and H, or initialize them
    W, H = _initialize_nmf(X, n_components, init=init, random_state=random_state)

    W, H, n_iter = _fit_coordinate_descent(X=X, W=W, H=H, tol=tol, max_iter=max_iter,
                                           l2_reg_W=alpha, l2_reg_H=alpha,
                                           verbose=verbose, shuffle=shuffle,
                                           random_state=random_state)

    if n_iter == max_iter and tol > 0:
        warnings.warn("Maximum number of iteration %d reached. Increase it to"
                      " improve convergence." % max_iter, ConvergenceWarning)

    return W, H, n_iter
