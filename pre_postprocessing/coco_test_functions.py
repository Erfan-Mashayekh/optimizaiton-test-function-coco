#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wen Jun 30 14:51:33 2021

Functions mostly from: https://www.sfu.ca/~ssurjano/index.html

@author: Lisa Pretsch
"""
import numpy as np
from math import exp, pi

class TestFunction:
    # Instance attributes
    def __init__(self, name=None, dim=None, lb=None, ub=None):
        self.name = name        # name
        self.dim = dim          # dimensionality, number of design variables
        self.lb = lb            # lower bound of design variables as numpy array
        self.ub = ub            # upper bound of design variables as numpy array

    # Instance methods
    def __str__(self):
        return f'Test function: {self.name}, {self.dim}-dimensional, lower bound {self.lb}, upper bound {self.ub}.'

    def evaluate(self, x):
         print(f'Error: Method "evaluate" for {self.name} is not yet implemented.')


class COCOTestFunction(TestFunction):
    """
    Usually evaluated on [-5,5]^d
    Global minimum at x_i=1 where f(x)=0
    """
    def __init__(self, id, constr, dim, xopt, fopt):
        super().__init__(
            name='F1',
            dim=dim,
            lb=-5*np.ones(dim),
            ub=5*np.ones(dim))

        self.id = id
        self.constr = constr
        self.xopt = xopt        # location of optimal x
        self.fopt = fopt        # optimal value        


    def tosz(self,x):
        """
        Non-Linear Transformation
        Rn → Rn,  for any positive integer n (n = 1 and n = D are used in the following), maps element-wise
        """
        xhat = np.where(x!=0, np.log(abs(x)), 0)
        c1 = np.where(x>0, 7.9, 3.1)
        c2 = np.where(x>0, 10.0, 5.5)

        return np.sign(x) * np.exp( xhat + 0.049 * (np.sin(c1 * xhat) + np.cos(c2 * xhat) ) )


    def evaluate(self, x):

        xopt = self.xopt[0] * np.ones(x.shape)[0::]

        if self.id == 1:     # F1            
            z = x - xopt
            f = np.sum(z ** 2, axis=0) + self.fopt
            return f    

        elif self.id == 2:   # F2
            tosz = self.tosz(x)
            z = tosz * (x - xopt)
            print(z.shape)
            f = np.sum()
            return 
