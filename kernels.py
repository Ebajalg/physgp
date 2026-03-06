import numpy as np
from sympy import *
from sklearn.gaussian_process.kernels import StationaryKernelMixin, NormalizedKernelMixin, Hyperparameter, Kernel
from time import time
from inspect import signature
import re


class PhysKerBase:
    def __init__(self, prior_kernel, lin_op, boundary_operators=None, boundary_conditions=None):
        self.prior_kernel = prior_kernel
        self.lin_op = lin_op
        
        # Extracting params in linear operator 
        self.lin_op_params = [i for i in tuple(self.lin_op.free_symbols) if not (str(i).startswith("x") or str(i).startswith("u"))]

        self.params_to_specify = [i for i in tuple(self.lin_op.free_symbols)+tuple(self.prior_kernel.u_func.free_symbols) 
                                  if not (str(i).startswith("x") or str(i).startswith("u") or str(i).startswith("y"))]
        for boundary_operator in boundary_operators:
            boundary_params = [i for i in tuple(boundary_operator.free_symbols)
                                  if not (str(i).startswith("x") or str(i).startswith("u") or str(i).startswith("y"))]
            self.params_to_specify = self.params_to_specify + boundary_params

        # Specifying boundary
        self.boundary_operators = boundary_operators
        if boundary_conditions is None:
            self.boundary_conditions = lambda *x_vals : "u"
        else:
            self.boundary_conditions = lambda *x_vals : sorted(['b'+str(i) for i,cond in enumerate(boundary_conditions) if cond(*x_vals)]+["u"])

        self.train_X = None
        
           
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

        self.k_fu = self.lin_op.subs({**{"u":self.k_uu}, **label_param_function(1)}).doit()
        kernel_func_dict['fu'] = self.k_fu

        self.k_uf = self.lin_op.subs({**{f"x{i}":f"y{i}" for i in range(self.prior_kernel.dim)},
                                 **label_param_function(0)}).subs({"u":self.k_uu}).doit()
        kernel_func_dict['uf'] = self.k_uf

        self.k_ff = self.lin_op.subs({**{"u":self.k_uf}, **label_param_function(1)}).doit()
        kernel_func_dict['ff'] = self.k_ff
        
        
        # Constructing functions that interact with boundary operators
        if self.boundary_operators is not None:
            for i,B in enumerate(self.boundary_operators):
                self.k_bu = B.subs({"u":self.k_uu}).doit()
                kernel_func_dict[f"b{i}u"] = self.k_bu

                self.k_ub = B.subs({f"x{i}":f"y{i}" for i in range(self.prior_kernel.dim)}).subs({"u":self.k_uu}).doit()
                kernel_func_dict[f"ub{i}"] = self.k_ub
                
                self.k_bf = B.subs({"u":self.k_uf}).doit()
                kernel_func_dict[f"b{i}f"] = self.k_bf

                self.k_fb = B.subs({f"x{i}":f"y{i}" for i in range(self.prior_kernel.dim)}).subs({"u":self.k_fu}).doit()
                kernel_func_dict[f"fb{i}"] = self.k_fb
                
                for j,D in enumerate(self.boundary_operators):
                    self.k_bd = D.subs({f"x{i}":f"y{i}" for i in range(self.prior_kernel.dim)}).subs({"u":self.k_bu}).doit()
                    kernel_func_dict[f"b{i}b{j}"] = self.k_bd
                
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
        if self.train_X is None:
            self.train_X = X


        def label_vec_func(X_u, X_f):
            label_vec = []
            for i,u_point in enumerate(X_u):
                label_list = self.boundary_conditions(u_point)
                if sum(X_u == u_point) > 1:
                    index = np.where(i == np.where(X_u == u_point)[0])[0][0]
                    label = label_list[index]
                else:
                    label = label_list[0]
                label_vec.append(label)
        
            label_vec += ['f' for i in range(X_f.shape[0])]

            return label_vec


        X_u = X[0]
        X_f = X[1]

        if Y is None:
            Y = X

        Y_u = Y[0]
        Y_f = Y[1]

        label_both = False
        if X_u.shape == Y_u.shape:
            if X_u.shape[0] == Y_u.shape[0] and np.allclose(X_u, Y_u):
                label_both = True
                
                
        if label_both:
            if X_u.shape[0] == self.train_X[0].shape[0]:    
                label_vec1 = label_vec_func(X_u, X_f)
                label_vec2 = label_vec_func(Y_u, Y_f)
            else:
                label_vec1 = ['u' for i in range(X_u.shape[0])] + ['f' for i in range(X_f.shape[0])]
                label_vec2 = ['u' for i in range(Y_u.shape[0])] + ['f' for i in range(Y_f.shape[0])]
        else:
            label_vec2 = label_vec_func(Y_u, Y_f)
            label_vec1 = ['u' for i in range(X_u.shape[0])] + ['f' for i in range(X_f.shape[0])]


        label_mat = np.matrix([[label1+label2 for label2 in label_vec2] for label1 in label_vec1])
    
        return label_mat
    
    
    
    def K_func(self, X, Y=None):
        """
        For X and Y data given, defines the function K_mat_func.
        
        :param X: Tuple of points in the domain of u and f, (X_u, X_f).
        :param Y: Tuple of points in the domain of u and f, (Y_u, Y_f), if different to X.
        :return: K_mat_func function.
        """
        
        label_mat = self.label_matrix(X, Y)
        
        X = np.vstack(X)

        if Y is not None:
            Y = np.vstack(Y)
            Y = Y
        else:
            Y = X
        
        def K_mat_func(param_dict, kernel_dict, sigma=(0,0)):
            """
            Creates the covariance matrix K based on the parameter values specified.
            If diff_param is not None, then creates the covariance matrix K differentiated 
            w.r.t. the given parameter.
            
            :param param_dict: Dictionary of parameter values.
            :param sigma: Tuple of noise values for u and f.
            :param diff_param: Symbol of parameter value to be differentiated.
            :return: covariance matrix K.
            """
                
            mat = np.zeros((X.shape[0], Y.shape[0]))
            
            dim_X = X.shape[1]
            dim_Y = Y.shape[1]

            for i in range(X.shape[0]):
                for j in range(Y.shape[0]):
                    x_dict = dict(zip([f"x{ind}" for ind in range(dim_X)], X[i, :]))
                    y_dict = dict(zip([f"y{ind}" for ind in range(dim_Y)], Y[j, :]))
                    
                    equ = kernel_dict[label_mat[i,j]]
                    arguments = re.sub("[() ]", "", str(signature(equ))).split(",")
                    if arguments == ['']:
                        value = 0
                    else:
                        value_dict = {k : {**x_dict, **y_dict, **param_dict}.get(k, None) for k in arguments}
                        value = equ(**value_dict)

                    if label_mat[i,j] in ['uu']:
                        value += sigma[0]**2
                    elif label_mat[i,j] in ['ff']:
                        value += sigma[1]**2

                    mat[i,j] = value
                    
            return mat

        return K_mat_func
    



class PhysKerGP(Kernel):
    def __init__(self, phys_kernel_base, param_values, param_bounds, sigma=(1e-3,1e-3)):
        phys_kernel_base.train_X = None
        self.phys_kernel_base = phys_kernel_base
        
        self.sigma = sigma
        self.param_values = param_values
        self.param_bounds = param_bounds
        self.param_names = param_values.keys()

        self.setup_kernel_dicts()
        self.lambdify_kernel_dict()

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
    

    def setup_kernel_dicts(self):
        self.kernel_dict = self.phys_kernel_base.kernel_dictionary()
        param_kernel_dict = {}
        for param in self.param_names:
            param_kernel_dict_iter = {key: kernel.diff(param) for key, kernel in self.kernel_dict.items()}
            param_kernel_dict[param] = param_kernel_dict_iter

        self.param_kernel_dict = param_kernel_dict


    def lambdify_kernel_dict(self):
        self.kernel_dict = {key: lambdify(tuple(kernel.free_symbols), kernel)
                             for key, kernel in self.kernel_dict.items()}
        
        self.param_kernel_dict = {param: {key: lambdify(tuple(kernel.free_symbols), kernel) for key, kernel in kernel_dict.items()}
                                   for param, kernel_dict in self.param_kernel_dict.items()}


    def __call__(self, X, Y=None, eval_gradient=False):
        X_u = X[X[:,0]==0][:,1:]
        X_f = X[X[:,0]==1][:,1:]

        X_fit = (X_u, X_f)

        if Y is None:
            Y_fit = X_fit
        else:
            Y_u = Y[Y[:,0]==0][:,1:]
            Y_f = Y[Y[:,0]==1][:,1:]
            Y_fit = (Y_u, Y_f)

        K_func = self.phys_kernel_base.K_func(X_fit, Y_fit)
        K = K_func(self.param_values, self.kernel_dict, sigma=self.sigma)

        if eval_gradient:
            K_gradient = np.zeros((K.shape[0], K.shape[1], len(self.param_values)))
            for i, param in enumerate(self.param_names):
                K_gradient[:,:,i] = K_func(self.param_values, kernel_dict=self.param_kernel_dict[param])
            return K, K_gradient
        else:
            return K
        


    def diag(self, X):
        return np.ones(X.shape[0])

    def is_stationary(self):
        return False