import numpy as np
import scipy.special as sp


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



def gen_point_grid(N, bounds, type_sample="uniform"):
    """
    Generates a grid of points within the hypercube definied
    by the tensor product of bounds. 

    :param N: int or list - specifies how many points should be in each
              1-d subspace of the hypercube.
    :param bounds: list of tuples - defines the bounds for each subspace,
                   tensor product of the bounds forms the hypercube.
    :param type_sample: str, int or list - defines the way in which the points are
                        sampled in each subspace, can either be one of 
                        ['uniform', 'chebyshev', 'legendre', 'random'] or an number corresponding
                        to the value for the lambda parameter for a
                        ultraspherical polynomial whose roots are used to sample the subspace.
    """

    # Get dimensions for hypercube.
    dim = len(bounds)

    # Check if only one N has been provided.
    if isinstance(N, int):
        N = np.repeat(N, dim)

    # Check if only one type_sample has been provided.
    if isinstance(type_sample, str):
        type_sample = np.repeat(type_sample, dim)
    if isinstance(type_sample, int):
        type_sample = np.repeat(type_sample, dim)

    # Rescale function for if include_boundary is set to True
    rescale = lambda a,b,vec : (a+b)/2 + ((b-a)/2)*vec

    points_list = []
    # Loop through bounds and N values
    for i, n, bound in enumerate(zip(N, bounds)):

        # Check type_sample and sample the subspace.
        if type_sample[i] == "uniform":
            points = np.linspace(bound[0], bound[1], n)

        if type_sample[i] == "chebyshev":
            points = rescale(bound[0], bound[1], sp.roots_gegenbauer(n, 1)[0])

        if type_sample[i] == "legendre":
            points = rescale(bound[0], bound[1], sp.roots_gegenbauer(n, 0.5)[0])


        if type_sample[i] == "legendre":
            points = np.random.normal(bound[0], bound[1], n)

        if isinstance(type_sample[i],int):
            points = rescale(bound[0], bound[1], sp.roots_gegenbauer(n, type_sample[i])[0])


        else:
            # Raise error if incorrect type_sample has been provided.
            raise ValueError("Please specify type as either a lambda value or either " \
            "'uniform', 'chebyshev' or 'legendre', or a list combination of them.")
        
        points_list.append(points)
        
    # Form hypercube grid.
    point_grid = np.meshgrid(tuple(points_list))

    # Convert from grid to vector of points.
    point_vector = np.vstack((ps.ravel() for ps in point_grid)).T

    return point_vector

        
