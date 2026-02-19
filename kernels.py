import numpy as np
from sympy import *
from sklearn.gaussian_process.kernels import StationaryKernelMixin, NormalizedKernelMixin, Hyperparameter, Kernel



class PhysKerBase:
    def __init__(self, prior_kernel, lin_op, boundary_operators=None, boundary_conditions=None):
        self.prior_kernel = prior_kernel
        self.lin_op = lin_op
        
        # Extracting params in linear operator 
        self.lin_op_params = [i for i in tuple(self.lin_op.free_symbols) if not (str(i).startswith("x") or str(i).startswith("u"))]

        # Specifying boundary
        self.boundary_operators = boundary_operators
        if boundary_conditions is None:
            self.boundary_conditions = lambda *x_vals : "u"
        else:
            self.boundary_conditions = lambda *x_vals : min(['b'+str(i) for i,cond in enumerate(boundary_conditions) if cond(*x_vals)]+["u"])
        
        
        
    def kernel_dictionary(self, inverse_problem_param=None):
        """
        Builds a dictionary with all kernel functions and builds expressions for them. 
        E.g. K_ff = L (u_kernel_prior) L'
        This is then used to know which kernel function to use with a pair of values.
        
        :return: Dictionary with all kernel functions (keys are short labels 'ff' for k_ff).
        """
        
        # Building dictionary that links pointer to sympy function
        kernel_func_dict = {}
        if inverse_problem_param is None:
            label_param_function = lambda i : {}
        else:
            label_param_function = lambda i : {inverse_problem_param:inverse_problem_param+str(i)} 
        

        # Constructing functions that interact with only the linear operator
        self.k_uu = self.prior_kernel.u_func
        kernel_func_dict['uu'] = self.k_uu

        self.k_uf = self.lin_op.subs({**{"u":self.k_uu}, **label_param_function(1)}).doit()
        kernel_func_dict['uf'] = self.k_uf

        self.k_fu = self.lin_op.subs({**{f"x{i}":f"y{i}" for i in range(self.prior_kernel.dim)},
                                 **label_param_function(0)}).subs({"u":self.k_uu}).doit()
        kernel_func_dict['fu'] = self.k_fu

        self.k_ff = self.lin_op.subs({**{f"x{i}":f"y{i}" for i in range(self.prior_kernel.dim)},
                                 **label_param_function(0)}).subs({"u":self.k_uf}).doit()
        kernel_func_dict['ff'] = self.k_ff
        
        
        # Constructing functions that interact with boundary operators
        if self.boundary_operators is not None:
            for i,B in enumerate(self.boundary_operators):
                self.k_ub = B.subs({"u":self.k_uu}).doit()
                kernel_func_dict[f"ub{i}"] = self.k_ub

                self.k_bu = B.subs({f"x{i}":f"y{i}" for i in range(self.prior_kernel.dim)}).subs({"u":self.k_uu}).doit()
                kernel_func_dict[f"b{i}u"] = self.k_bu
                
                self.k_fb = B.subs({"u":self.k_uf}).doit()
                kernel_func_dict[f"fb{i}"] = self.k_fb

                self.k_bf = B.subs({f"x{i}":f"y{i}" for i in range(self.prior_kernel.dim)}).subs({"u":self.k_uf}).doit()
                kernel_func_dict[f"b{i}f"] = self.k_bf
                
                for j,D in enumerate(self.boundary_operators):
                    self.k_db = D.subs({f"x{i}":f"y{i}" for i in range(self.prior_kernel.dim)}).subs({"u":self.k_ub}).doit()
                    kernel_func_dict[f"b{j}b{i}"] = self.k_db
                
        return kernel_func_dict
    
    
    
    def label_matrix(self, X, Y=None):
        """
        For data given, provides the labels for which kernel function should be used
        for the pair (x_i,y_j). Automatically identifies points within the domain of u
        that lie on the specified boundary (defined by boundary_conditions).
        
        :param X: Tuple of points in the domain of u and f, (X_u, X_f).
        :param Y: Tuple of points in the domain of u and f, (Y_u, Y_f), if different to X.
        :return: Matrix with labels for which kernel function should be used.
        """
        
        X_u = X[0]
        X_f = X[1]
    
        label_vec1 = []
        for u_point in X_u:
            label = self.boundary_conditions(*u_point)
            label_vec1.append(label)
    
        label_vec1 += ['f' for i in range(X_f.shape[0])]
    
        if Y is not None:
            Y_u = Y[0]
            Y_f = Y[1]
    
            label_vec2 = []
            for u_point in Y_u:
                label = self.boundary_conditions(*u_point)
                label_vec2.append(label)
    
            label_vec2 += ['f' for i in range(Y_f.shape[0])]
        else:
            label_vec2 = label_vec1

        label_mat = np.matrix([[label1+label2 for label2 in label_vec1] for label1 in label_vec2])
    
        return label_mat
    
    
    
    def K_func(self, X, Y=None):
        """
        For X and Y data given, defines the function K_mat_func.
        
        :param X: Tuple of points in the domain of u and f, (X_u, X_f).
        :param Y: Tuple of points in the domain of u and f, (Y_u, Y_f), if different to X.
        :return: K_mat_func function.
        """
        
        label_mat = self.label_matrix(X, Y)
        
        X = np.hstack(X)

        if Y is not None:
            Y = np.hstack(Y)
            Y = Y
        else:
            Y = X
        
        
        def K_mat_func(param_dict, sigma=(0,0), diff_param=None, inverse_problem_param=None):
            """
            Creates the covariance matrix K based on the parameter values specified.
            If diff_param is not None, then creates the covariance matrix K differentiated 
            w.r.t. the given parameter.
            
            :param param_dict: Dictionary of parameter values.
            :param sigma: Tuple of noise values for u and f.
            :param diff_param: Symbol of parameter value to be differentiated.
            :return: covariance matrix K.
            """
            
            kernel_dict_ = self.kernel_dictionary(inverse_problem_param=inverse_problem_param)

            if diff_param is None:
                kernel_dict = kernel_dict_
            else:
                kernel_dict = {}
                for key, equ in kernel_dict_.items():
                    diff_equ = equ.diff(diff_param)
                    kernel_dict[key] = diff_equ
                
            print(X)
            print(Y)

            mat = np.zeros((X.shape[1], Y.shape[1]))
            
            dim_X = X.shape[0]
            dim_Y = Y.shape[0]
            
            for i in range(X.shape[1]):
                for j in range(Y.shape[1]):
                    x_dict = dict(zip([f"x{ind}" for ind in range(dim_X)], X[:, i]))
                    y_dict = dict(zip([f"y{ind}" for ind in range(dim_Y)], Y[:, j]))
                    
                    equ = kernel_dict[label_mat[i,j]]
                    value = equ.evalf(subs={**x_dict, **y_dict, **param_dict})

                    if label_mat[i,j] in ['uu']:
                        value += sigma[0]**2
                    elif label_mat[i,j] in ['ff']:
                        value += sigma[1]**2

                    mat[i,j] = value
                    
            return mat

        return K_mat_func
    



class PhysKerGP(Kernel):
    def __init__(self, phys_kernel_base, param_values, param_bounds, sigma=(1e-3,1e-3)):
        self.phys_kernel_base = phys_kernel_base
        
        self.sigma = sigma
        self.param_values = param_values
        self.param_bounds = param_bounds
        self.param_names = param_values.keys()

        self.setup_params()


    def setup_params(self):
        for param, val in self.param_values.items():
            setattr(self.__class__, "hyperparameter_"+param,
                    property(lambda s: Hyperparameter(param, 'numeric', s.param_bounds[param])))
    
    @property
    def theta(self):
        return np.array(list(self.param_values.values())).T
    
    @theta.setter
    def theta(self, theta):
        keys = self.param_values.keys()
        self.param_values = dict(zip(keys, list(theta.T)))

    @property
    def bounds(self):
        return np.vstack(list(self.param_bounds.values()))
    


    def __call__(self, X, Y=None, eval_gradient=False):
        X_u = X[X[:,0]==0][:,1:].T
        X_f = X[X[:,0]==1][:,1:].T

        X_fit = (X_u, X_f)

        if Y is None:
            Y_fit = X_fit
        else:
            Y_u = Y[Y[:,0]==0][:,1:].T
            Y_f = Y[Y[:,0]==1][:,1:].T
            Y_fit = (Y_u, Y_f)

        K_func = self.phys_kernel_base.K_func(X_fit, Y_fit)
        K = K_func(self.param_values, sigma=self.sigma)

        print(K)
        print(K.shape)

        if eval_gradient:
            K_gradient = np.zeros((K.shape[0], K.shape[1], len(self.param_values)))
            for i, param in enumerate(self.param_names):
                print(f"{i}-th parameter: {param}")
                K_gradient[:,:,i] = K_func(self.param_values, diff_param=param)
            return K, K_gradient
        else:
            return K
        


    def diag(self, X):
        return np.ones(X.shape[0])

    def is_stationary(self):
        return False