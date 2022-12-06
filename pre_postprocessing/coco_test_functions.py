#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wen Jun 30 14:51:33 2021

Functions mostly from: https://www.sfu.ca/~ssurjano/index.html

@author: Lisa Pretsch
"""
import numpy as np


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
    def __init__(self, id, constrained, constr_number, dim, xopt, fopt, ystar):
        super().__init__(
            name='F1',
            dim=dim,
            lb=-10*np.ones(dim),
            ub=10*np.ones(dim))

        self.id = id
        self.xopt = xopt        # location of optimal point
        self.fopt = fopt        # optimal value        
        self.ystar = ystar      # location of constrained optimal point
        self.constrained = constrained
        self.constr_number = constr_number
        self.a = np.zeros((self.constr_number, self.dim))


    def tosz(self,x):
        """
        Non-Linear Transformation
        Rn → Rn,  for any positive integer n (n = 1 and n = dim are used in the following), maps element-wise
        """
        xhat = np.where(x!=0, np.log(abs(x)), 0)
        c1 = np.where(x>0, 7.9, 3.1)
        c2 = np.where(x>0, 10.0, 5.5)

        return np.sign(x) * np.exp( xhat + 0.049 * (np.sin(c1 * xhat) + np.cos(c2 * xhat) ) )


    def evaluate(self, x):

        xopt = self.xopt[0] * np.ones(x.shape)[0]

        if self.id == 1:     # F1: Sphere Function      
            z = x - xopt
            f = np.sum(z ** 2, axis=0) + self.fopt
            return f    

        elif self.id == 2:   # F2: Separable Ellipsoidal Function
            z = self.tosz(x - xopt)

            i = (np.arange(self.dim) + 1)
            if np.size(x.shape) > 1:
                shape = np.ones(np.size(x.shape))
                shape[0] = self.dim
                i = np.broadcast_to(i.reshape(shape.astype(int)), x.shape)
            f = np.sum(10 ** (6 * (i-1) / (self.dim-1)) * z ** 2, axis=0) + self.fopt

            return f

        elif self.id == 5:   # F5: Linear Slop
            z = np.where(xopt*x < 5**2, x, xopt)
            i = (np.arange(self.dim) + 1)
            if np.size(x.shape) > 1:
                shape = np.ones(np.size(x.shape))
                shape[0] = self.dim
                i = np.broadcast_to(i.reshape(shape.astype(int)), x.shape)
            s = np.sign(xopt) * 10 ** ((i-1) / (self.dim-1))
            f = np.sum(5 * abs(s) - s * z, axis=0) + self.fopt

            return f

    def grid(self):
        x1 = np.linspace(self.lb[0], self.ub[0], 300)
        x2 = np.linspace(self.lb[1], self.ub[1], 300)
        x1_mesh, x2_mesh = np.meshgrid(x1, x2)
        inputs = np.array([x1_mesh, x2_mesh])
        f = self.evaluate(inputs)
        return x1_mesh, x2_mesh, f

    def compute_a(self, alpha, grid):

        self.a = np.zeros((self.constr_number, self.dim))

        x1 = grid[0]
        x2 = grid[1]
        f = grid[2]
        f_shape = list(f.shape)
        f_shape.insert(0, self.dim)
        grad_f = np.zeros(tuple(f_shape))

        # TODO: check gradient computation
        for i in range(self.dim):
            grad_f[i,:] = np.gradient(f, axis=(self.dim-i-1))

        #TODO: find the location of the y_star in grad_f
        x1_idx = (np.abs(x1[0,:] - self.ystar[0])).argmin()
        x2_idx = (np.abs(x2[:,0] - self.ystar[1])).argmin()
        grad_f_ystar = grad_f[:,x2_idx,x1_idx]
        self.a[0,:] = - alpha * grad_f_ystar / np.linalg.norm(grad_f_ystar)
        self.a[1:,:] = np.random.normal(self.a[0,:], 0.8, size=(self.constr_number-1,self.dim))

    def initialize_constraint_parameters(self):
        alpha = np.ones(self.constr_number)
        b = np.zeros(self.constr_number)
        self.compute_a(alpha[0], self.grid())

    def compute_g(self, x):
        #g = alpha * (a @ (x-self.ystar).T) + b
        g = (self.a @ (x-self.ystar).T)
        g[1] = g[1] + 2.
        return -g