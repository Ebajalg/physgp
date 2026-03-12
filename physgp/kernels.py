import numpy as np
from sympy import *
from sklearn.gaussian_process.kernels import StationaryKernelMixin, NormalizedKernelMixin, Hyperparameter, Kernel
from inspect import signature
import re


class PhysKerBase:
    def __init__(self, prior_kernel, lin_op, boundary_operators=None, boundary_conditions=None):
        """
        Base kernel object for Physics-Informed gaussian process regression. All of the specifcation of the PDE problem is made.
        Used to construct all of the required, covariance functions.

        :param prior_kernel: A prior kernel object.
        :param lin_op: A sympy expression for the linear operator.
        :param boundary_operators: List of sympy expressions (each representing a different boundary operator).
        :param boundary_conditions: List of lambda functions that return a Boolean value for if the point is on the boundary
                                    where the corresponding boundary conditions is specified. 
                                    (the i th condition corresponds to the i th boundary operator)
        """

        self.prior_kernel = prior_kernel
        self.lin_op = lin_op
        
        # Extracting params in linear operator.
        self.lin_op_params = [i for i in tuple(self.lin_op.free_symbols) if not (str(i).startswith("x") or str(i).startswith("u"))]

        # Extracting all hyperparameters that need to be specified.
        self.params_to_specify = [i for i in tuple(self.prior_kernel.u_func.free_symbols) 
                                  if not (str(i).startswith("x") or str(i).startswith("u") or str(i).startswith("y"))] + self.lin_op_params
        
        # Adding the parameters that appear within the boundary operators.
        if not boundary_operators is None:
            for boundary_operator in boundary_operators:
                boundary_params = [i for i in tuple(boundary_operator.free_symbols)
                                    if not (str(i).startswith("x") or str(i).startswith("u") or str(i).startswith("y"))]
                self.params_to_specify = self.params_to_specify + boundary_params

        # Specifying boundary.
        self.boundary_operators = boundary_operators

        # Building lambda function that returns a list of all boundary conditions satisified by a point x.
        if boundary_conditions is None:
            self.boundary_conditions = lambda *x_vals : "u"
        else:
            self.boundary_conditions = lambda *x_vals : sorted(['b'+str(i) for i,cond in enumerate(boundary_conditions) if cond(*x_vals)]+["u"])

        # Used to keep track on when the kernel is being fitted or used to predict with.
        self.train_X = None

        # Generates all of the necessary expressions for the physics-informed gp kernel.
        self.kernel_expression_dict = self.kernel_dictionary()
        
           

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
    

    
    def label_vec_func(self, X_u, X_f):
        """
        Labels the full vector [X_u  X_f] with boundary conditions considered.

        :param X_u: numpy array of the X values for the observations of u.
        :param X_f: numpy array of the X values for the observations of f.
        """

        label_vec = []

        # Iterating through the u observations
        for i,u_point in enumerate(X_u):

            # Identifying all the satisfied boundary conditions.
            label_list = self.boundary_conditions(u_point)

            # If none are satisfied, defaults to label as u.
            if len(label_list) == 1:
                label_vec.append(label_list[0])
                continue
            
            # Checks if there are multiple instances of the x point.
            if sum(np.all(X_u == u_point, axis=1)) > 1:
                
                # Identifies how many of those points have already been labelled.
                # Means that each instance of the point will be labelled with a different boundary
                # the order of this labelling is specified by the ordering of the boundary conditions.

                index = np.where(i == np.where(np.all(X_u == u_point, axis=1))[0])[0][0]
                # Labels with the next point in the label_list

                if index+1 > len(label_list):
                    # If all boundary conditions for the duplicate point have already been specified 
                    # default to u
                    label = "u"
                else:
                    # Else label the point with the corresponding b
                    label = label_list[index]
            else:
                label = label_list[0]

            # Appending label to the list of labels
            label_vec.append(label)
        
        # Labelling all of the instances of f
        label_vec += ['f' for i in range(X_f.shape[0])]

        return label_vec
    
    
    def label_matrix(self, X, Y=None):
        """
        For data given, provides the labels for which kernel function should be used
        for the pair (x_i,y_j). Automatically identifies points within the domain of u
        that lie on the specified boundary (defined by boundary_conditions).
        
        :param X: Tuple of points in the domain of u and f, (X_u, X_f).
        :param Y: Tuple of points in the domain of u and f, (Y_u, Y_f), if different to X.
        :return: Matrix with labels for which kernel function should be used.
        """

        # Checks if training has been completed (model fitting),
        # If not it sets the training set so it can be checked against when it comes to prediction.
        if self.train_X is None:
            self.train_X = X

        # Splitting X to X_u and X_f
        X_u = X[0]
        X_f = X[1]

        # Checking Y
        if Y is None:
            Y = X

        Y_u = Y[0]
        Y_f = Y[1]

        label_both = False

        # If the vectors are the same, modelling fitting is assumed
        # boundary factors are factored into the labelling.
        if X_u.shape == Y_u.shape:
            if X_u.shape[0] == Y_u.shape[0] and np.allclose(X_u, Y_u):
                label_both = True
                
        
        if label_both:
            # If the training data against the training data is being labelled, label with BCs.
            if X_u.shape[0] == self.train_X[0].shape[0]:    
                label_vec1 = self.label_vec_func(X_u, X_f)
                label_vec2 = self.label_vec_func(Y_u, Y_f)
            else:
                # If not then K_uu or K_ff is being made for prediction
                # Hence BCs labels are not included.
                label_vec1 = ['u' for i in range(X_u.shape[0])] + ['f' for i in range(X_f.shape[0])]
                label_vec2 = ['u' for i in range(Y_u.shape[0])] + ['f' for i in range(Y_f.shape[0])]
        else:
            # Labelling without BCs for prediction data points
            # With BCs for the training data points.
            label_vec2 = self.label_vec_func(Y_u, Y_f)
            label_vec1 = ['u' for i in range(X_u.shape[0])] + ['f' for i in range(X_f.shape[0])]

        # Taking the two label vectors and forming the grid from the cross of them.
        # i.e. if one [u,b0,f] and the other [u,u,f], then the label matrix will be.
        #  
        # | uu ub0 uf |
        # | uu ub0 uf |
        # | fu fb0 ff |

        label_mat = np.matrix([[label1+label2 for label2 in label_vec2] for label1 in label_vec1])
    
        return label_mat
    
    
    
    def K_func(self, X, Y=None):
        """
        For X and Y data given, defines the function K_mat_func.
        
        :param X: Tuple of points in the domain of u and f, (X_u, X_f).
        :param Y: Tuple of points in the domain of u and f, (Y_u, Y_f), if different to X.
        :return: K_mat_func function.
        """
        
        # Forming label matrix.
        label_mat = self.label_matrix(X, Y)
        
        # Forming X and Y data for K_mat_function.
        X = np.vstack(X)

        if Y is not None:
            Y = np.vstack(Y)
            Y = Y
        else:
            Y = X
        
        def K_mat_func(param_dict, kernel_dict, sigma=(0,0)):
            """
            Creates the covariance matrix K based on the parameter values specified
            and the kernel function dictionary (which can change if we are considering the
            hyperparameter derivative kernels).
            
            :param param_dict: Dictionary of parameter values.
            :param kernel_dict: Dictionary of lambda functions for the corresponding kernels.
            :param sigma: Tuple of noise values for u and f (repectively).
            """
                
            # Empty matrix
            mat = np.zeros((X.shape[0], Y.shape[0]))
            
            dim_X = X.shape[1]
            dim_Y = Y.shape[1]

            # Looping through entries of empty matrix.
            for i in range(X.shape[0]):
                for j in range(Y.shape[0]):

                    # Assigning the correct values for x and y via a dictionary.
                    x_dict = dict(zip([f"x{ind}" for ind in range(dim_X)], X[i, :]))
                    y_dict = dict(zip([f"y{ind}" for ind in range(dim_Y)], Y[j, :]))
                    
                    # Getting the lambda function.
                    equ = kernel_dict[label_mat[i,j]]

                    # Getting the arguements of the lambda function.
                    arguments = re.sub("[() ]", "", str(signature(equ))).split(",")

                    if arguments == ['']:
                        # If no arguments are given the function will be 0.
                        value = 0
                    else:
                        # Label all of the agruments for the equation.
                        value_dict = {k : {**x_dict, **y_dict, **param_dict}.get(k, None) for k in arguments}
                        # Calculate value for the kernel.
                        value = equ(**value_dict)

                    # Adding noise, only in the case of interactions between,
                    # u and u or f and f.
                    if label_mat[i,j] in ['uu']:
                        value += sigma[0]**2
                    elif label_mat[i,j] in ['ff']:
                        value += sigma[1]**2

                    # Assigning the value.
                    mat[i,j] = value
                    
            return mat

        return K_mat_func
    







class PhysKerGP(Kernel):
    def __init__(self, phys_kernel_base, param_values, param_bounds, sigma=(1e-15,1e-15)):
        """
        This object acts as a wrapper for the PhysKerBase object, in order for it to act
        as a kernel that sckit-learn's GP regressor can accept.

        :param phys_kernel_base: PhysKerBase object.
        :param param_values: Dictionary of the hyperparameter values.
        :param param_bounds: Dictionary of the bounds for each of the hyperparameters.
        :param sigma: Tuple of noise values for u and f (repectively).
        """

        # Makes sure the train_X is None.
        phys_kernel_base.train_X = None
        self.phys_kernel_base = phys_kernel_base
        
        self.sigma = sigma
        self.param_values = param_values
        self.param_bounds = param_bounds
        self.param_names = param_values.keys()

        # Builds the kernels and the hyperparameter paritial derviatives for the kernels.
        self.setup_kernel_dicts()
        # Takes the sympy expression and lambdaifies them.
        self.lambdify_kernel_dict()

        # Sets up the parameters as individual Hyperparameter objects.
        self.setup_params()


    # Sets up all of the hyperparameters with the correct property name formatting.
    def setup_params(self):
        for param, val in self.param_values.items():
            setattr(self.__class__, "hyperparameter_"+param,
                    property(lambda s: Hyperparameter(param, 'numeric', s.param_bounds[param])))
    

    # Property required to access current values for the hyperparameters.
    @property
    def theta(self):
        return np.array(list(self.param_values.values())).T
    

    # Property required to update the values for the hyperparameters.
    @theta.setter
    def theta(self, theta):
        keys = self.param_values.keys()
        self.param_values = dict(zip(keys, list(theta.T)))


    # Property to access the bounds for the hyperparameters.
    @property
    def bounds(self):
        return np.vstack(list(self.param_bounds.values()))
    

    def setup_kernel_dicts(self):
        # Fetches the kernel dictionary 
        self.kernel_dict = self.phys_kernel_base.kernel_expression_dict

        # Calculating the partial derivatives w.r.t to the hyperparameters.
        param_kernel_dict = {}
        for param in self.param_names:
            param_kernel_dict_iter = {key: kernel.diff(param) for key, kernel in self.kernel_dict.items()}
            param_kernel_dict[param] = param_kernel_dict_iter

        self.param_kernel_dict = param_kernel_dict


    def lambdify_kernel_dict(self):
        # Going through all of the sympy expressions and lambdifying them.
        self.kernel_dict = {key: lambdify(tuple(kernel.free_symbols), kernel)
                             for key, kernel in self.kernel_dict.items()}
        
        self.param_kernel_dict = {param: {key: lambdify(tuple(kernel.free_symbols), kernel) for key, kernel in kernel_dict.items()}
                                   for param, kernel_dict in self.param_kernel_dict.items()}



    def __call__(self, X, Y=None, eval_gradient=False):
        """
        Forms the covariance matrix and the hyperparameter partial derivatives 
        of the covariance matrix.

        :param X: X data.
        :param Y: Y data.
        :param eval_gradient: Boolean, controls if the partial derivative matrices are calculated.
        """

        # Retrieval of u and f observations based on labels for X.
        X_u = X[X[:,0]==0][:,1:]
        X_f = X[X[:,0]==1][:,1:]

        X_fit = (X_u, X_f)

        # Retrieval of u and f observations based on labels for Y.
        if Y is None:
            Y_fit = X_fit
        else:
            Y_u = Y[Y[:,0]==0][:,1:]
            Y_f = Y[Y[:,0]==1][:,1:]
            Y_fit = (Y_u, Y_f)


        # Builds the covariance matrix.
        K_func = self.phys_kernel_base.K_func(X_fit, Y_fit)
        K = K_func(self.param_values, self.kernel_dict, sigma=self.sigma)

        if eval_gradient:
            # Builds the partial derivatives of the covariance matrix w.r.t.
            # each hyperparameter.
            K_gradient = np.zeros((K.shape[0], K.shape[1], len(self.param_values)))

            for i, param in enumerate(self.param_names):
                K_gradient[:,:,i] = K_func(self.param_values, kernel_dict=self.param_kernel_dict[param])

            return K, K_gradient
        
        else:

            return K
        

    # These are unused but required by scikit-learn.
    def diag(self, X):
        return np.ones(X.shape[0])

    def is_stationary(self):
        return False