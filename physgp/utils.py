import numpy as np

def obs_concat(X_u=None, X_f=None):
    if (X_u is None) or (X_f is None):
        if (X_u is None) and (X_f is None):
            raise ValueError("No data provided.")
        if X_u is None:
            return np.column_stack((np.ones(X_f.shape[0]), X_f))
        if X_f is None:
            return np.column_stack((np.zeros(X_u.shape[0]), X_u))
    else:
        u_indexed = np.column_stack((np.zeros(X_u.shape[0]), X_u))
        f_indexed = np.column_stack((np.ones(X_f.shape[0]), X_f))

        return np.vstack((u_indexed, f_indexed))