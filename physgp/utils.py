import numpy as np

def obs_concat(X_u=None, X_f=None):
    """
    Concatinates and labels the u and f observations in order for the correct data to be
    passed into the GP regression function. X_u and X_f can be specfied without the other in
    the case of prediction.

    :param X_u: numpy array of the X values for the observations of u.
    :param X_f: numpy array of the X values for the observations of f.
    """

    # Checking for the case when only one of the sets of values has been specified.
    if (X_u is None) or (X_f is None):

        if (X_u is None) and (X_f is None):
            # Raise error if no data is provided.
            raise ValueError("No data provided.")
        
        if X_u is None:
            # Return X_f labelled.
            return np.column_stack((np.ones(X_f.shape[0]), X_f))
        
        if X_f is None:
            # Return X_u labelled.
            return np.column_stack((np.zeros(X_u.shape[0]), X_u))
        
    else:
        # Return X_u and X_f labelled and combined into one array.
        u_indexed = np.column_stack((np.zeros(X_u.shape[0]), X_u))
        f_indexed = np.column_stack((np.ones(X_f.shape[0]), X_f))

        return np.vstack((u_indexed, f_indexed))
    


def gen_gp_sample(m, K, alpha=1e-10):
    """
    Given a mean vector and a covariance matrix of a gaussian process,
    the function generates a sample from the gaussian process.

    :param m: numpy array of the mean vector of the gaussian process.
    :param K: numpy array/matrix for the covariance matrix of the gaussian process.
    :param alpha: noise added to the diagonal so the covariance matrix is computationally p.s.d.
    """

    N = m.shape[0]
    norm_rand = np.random.normal(0, 1, N) # Generating a random normal vector.

    # Calculating the square root of the covariance matrix.
    K_sqrt = np.linalg.cholesky(K+alpha*np.eye(N))

    # Returning the sampled function.
    # (realised at the points the mean vector was specified for)
    return m + K_sqrt @ norm_rand