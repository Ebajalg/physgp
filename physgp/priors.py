from sympy import *
import numpy as np

class PriorKernel:
    def __init__(self):
        self.u_func = None

    def __add__(self, other):
        u_func_self = str(self.u_func)
        u_func_other = str(other.u_func)
        u_func = sympify(u_func_self + "+" + u_func_other)

        # Implement check for overlap and raise error if detected.
        param_symbols = tuple(list(self.param_symbols) + list(other.param_symbols))

        dim = max(self.dim, other.dim)

        return CompoundKernel(u_func, dim, param_symbols)
    
    def __mul__(self, other):
        u_func_self = str(self.u_func)
        u_func_other = str(other.u_func)
        u_func = sympify(f"({u_func_self})*({u_func_other})")

        # Implement check for overlap and raise error if detected.
        param_symbols = tuple(list(self.param_symbols) + list(other.param_symbols))

        dim = max(self.dim, other.dim)

        return CompoundKernel(u_func, dim, param_symbols)



class CompoundKernel(PriorKernel):
    def __init__(self, u_func, dim, param_symbols):
        self.dim = dim
        self.param_symbols = param_symbols
        self.u_func = u_func





class SquExpKernel(PriorKernel):
    def __init__(self, dim, param_symbols=("w", "s")):
        self.dim_ = dim
        self.dim = max(dim) if isinstance(dim, list) else dim
        self.param_symbols = param_symbols
        self.data_symbols = ("x", "y")
        self.u_func = self.construct()

    def construct(self):
        expression_list = []

        if isinstance(self.dim_, list):
            for i in range(self.dim+1):
                if i in self.dim_: 
                    expression_list.append(f"{self.param_symbols[0]}{i}*({self.data_symbols[0]}{i}-{self.data_symbols[1]}{i})^2")
        else:
            for i in range(self.dim):
                expression_list.append(f"{self.param_symbols[0]}{i}*({self.data_symbols[0]}{i}-{self.data_symbols[1]}{i})^2")

        expression = " + ".join(expression_list)
        u_func = sympify(f"({self.param_symbols[1]}^2)*exp(-({expression})/2)")

        return u_func
    


class PeriodicKernel(PriorKernel):
    def __init__(self, dim, param_symbols=("l", "p", "s")):
        self.dim_ = dim
        self.dim = max(dim) if isinstance(dim, list) else dim
        self.param_symbols = param_symbols
        self.data_symbols = ("x", "y")
        self.u_func = self.construct()

    def construct(self):
        expression_list = []


        if isinstance(self.dim_, list):
            for i in range(self.dim+1):
                if i in self.dim_: 
                    expression_list.append(f"({self.data_symbols[0]}{i}-{self.data_symbols[1]}{i})")
        else:
            for i in range(self.dim): 
                expression_list.append(f"({self.data_symbols[0]}{i}-{self.data_symbols[1]}{i})")

        expression = " + ".join(expression_list)
        u_func = sympify(f"({self.param_symbols[2]}^2)*exp(-(2/{self.param_symbols[0]}^2)*sin({expression}*(pi/{self.param_symbols[1]}))**2)")

        return u_func
    


class LinearKernel(PriorKernel):
    def __init__(self, dim, param_symbols=("m", "c")):
        self.dim_ = dim
        self.dim = max(dim) if isinstance(dim, list) else dim
        self.param_symbols = param_symbols
        self.data_symbols = ("x", "y")
        self.u_func = self.construct()
    
    def construct(self):
        expression_list = []

        if isinstance(self.dim_, list):
            for i in range(self.dim+1):
                if i in self.dim_: 
                    expression_list.append(f"{self.data_symbols[0]}{i}*{self.data_symbols[1]}{i}")
        else:
            for i in range(self.dim): 
                expression_list.append(f"{self.data_symbols[0]}{i}*{self.data_symbols[1]}{i}")
        
        expression = " + ".join(expression_list)
        u_func = sympify(f"{self.param_symbols[0]}*({expression}) + {self.param_symbols[1]}")

        return u_func
    


class PolyKernel(PriorKernel):
    def __init__(self, dim, param_symbols=("m", "c", "k")):
        self.dim_ = dim
        self.dim = max(dim) if isinstance(dim, list) else dim
        self.param_symbols = param_symbols
        self.data_symbols = ("x", "y")
        self.u_func = self.construct()
    
    def construct(self):
        expression_list = []

        if isinstance(self.dim_, list):
            for i in range(self.dim+1):
                if i in self.dim_:  
                    expression_list.append(f"{self.data_symbols[0]}{i}*{self.data_symbols[1]}{i}")
        else:
            for i in range(self.dim): 
                expression_list.append(f"{self.data_symbols[0]}{i}*{self.data_symbols[1]}{i}")
        
        expression = " + ".join(expression_list)
        u_func = sympify(f"({self.param_symbols[0]}*({expression}) + {self.param_symbols[1]})^{self.param_symbols[2]}")

        return u_func
