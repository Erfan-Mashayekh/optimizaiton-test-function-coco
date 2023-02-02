# %%
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wen Jun 30 14:51:33 2021

Functions mostly from: https://www.sfu.ca/~ssurjano/index.html

@author: Lisa Pretsch
"""
import numpy as np

from ioh import problem
from ioh import get_problem, ProblemType



class COCOTestFunction:
    """
    Usually evaluated on [-5,5]^d
    Global minimum at x_i=1 where f(x)=0
    """

    def __init__(self, model, dim, constrained, num_constraints, lb, ub, xopt, fopt, ystar):

        self.coco_id = model['coco_id']  # name of test function
        self.dim = dim
        self.constrained = constrained
        self.num_constraints = num_constraints
        self.lb = lb
        self.ub = ub
        self.xopt = xopt
        self.fopt = fopt
        self.ystar = ystar
        
        self.a = np.zeros((self.num_constraints, self.dim))

    def evaluate(self, x): 
        f = get_problem(self.coco_id, 10, self.dim, ProblemType.BBOB)
        if x.size <= self.dim:
            return f(x)
        else:
            f_list = np.empty(x.shape[0])
            for i in range(x.shape[0]):
                f_list[i] = f(x[i]) 
            return f_list


    def grid(self, grid_size):
        """
        Generate an d-dimensional grid based on the parameter 'dim'
        :param grid_size: the size of the grid
        :return: x:
        """
        args = []
        for i in range(self.dim):
            args.append(np.linspace(self.lb[1], self.ub[1], grid_size))
        mesh = np.array(np.meshgrid(*args))
        mesh_reshaped = mesh.reshape((mesh.shape[0], mesh[0].size)).T
        f = self.evaluate(mesh_reshaped)

        return mesh, f

    def compute_a(self, alpha, grid, grid_size, constraints_seed):
        """
        Compute coefficient 'a' which helps to generate a random slope for a linear constraint.
        """
        self.a = np.zeros((self.num_constraints, self.dim))
        mesh, f = grid
        f = f.reshape(mesh[0].shape)
        grad_f = np.array(np.gradient(f.T))  # TODO: check if transpose works for higher dimensions
        grid_points = np.linspace(self.lb[1], self.ub[1], grid_size)
        #TODO: find y_star neighbors
        index = []
        for i in range(self.dim):
            index.append((np.abs(grid_points - self.ystar[i])).argmin())
        grad_f_ystar = np.zeros(self.dim)
        for i in range(self.dim):
            grad_f_ystar[i] = grad_f[i][tuple(index)]

        self.a[0, :] = - alpha * grad_f_ystar / np.linalg.norm(grad_f_ystar)
        np.random.seed(constraints_seed)
        self.a[1:, :] = np.random.normal(self.a[0, :], 0.8, size=(self.num_constraints - 1, self.dim))

    def initialize_constraint_parameters(self, grid_size, constraints_seed):
        alpha = np.ones(self.num_constraints)
        b = np.zeros(self.num_constraints)
        self.compute_a(alpha[0], self.grid(grid_size), grid_size, constraints_seed)
        return self.a

    def compute_g(self, x, a):
        # g = alpha * (a @ (x-self.ystar).T) + b
        g = (a @ (x - self.ystar).T)
        g[1] = g[1] + 2.
        return -g
