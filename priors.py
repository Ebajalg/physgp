from sympy import *
import numpy as np

class SquExpKernel:
    def __init__(self, dim):
        self.dim = dim
        self.u_func = self.construct()

    def construct(self):
        expression_list = []

        for i in range(self.dim):
            expression_list.append(f"w{i}*(x{i}-y{i})^2")

        expression = " + ".join(expression_list)
        u_func = sympify(f"(s^2)*exp(-({expression})/2)")

        return u_func