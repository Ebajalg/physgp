from sympy import *
import numpy as np

class SquExpKernel:
    def __init__(self, dim, param_symbols=("w", "s")):
        self.dim = dim
        self.param_symbols = param_symbols
        self.data_symbols = ("x", "y")
        self.u_func = self.construct()

    def construct(self):
        expression_list = []

        for i in range(self.dim):
            expression_list.append(f"{self.param_symbols[0]}{i}*({self.data_symbols[0]}{i}-{self.data_symbols[1]}{i})^2")

        expression = " + ".join(expression_list)
        u_func = sympify(f"({self.param_symbols[1]}^2)*exp(-({expression})/2)")

        return u_func
    

class PeriodicKernel:
    def __init__(self, dim, param_symbols=("l", "p", "s")):
        self.dim = dim
        self.param_symbols = param_symbols
        self.data_symbols = ("x", "y")
        self.u_func = self.construct()

    def construct(self):
        expression_list = []

        for i in range(self.dim): 
            expression_list.append(f"({self.data_symbols[0]}{i}-{self.data_symbols[1]}{i})")

        expression = " + ".join(expression_list)
        u_func = sympify(f"({self.param_symbols[2]}^2)*exp(-(2/{self.param_symbols[0]}^2)*sin({expression}*(pi/{self.param_symbols[1]}))**2)")

        return u_func
    

