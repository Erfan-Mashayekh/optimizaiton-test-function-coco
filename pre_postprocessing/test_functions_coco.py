# %%
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wen Jun 30 14:51:33 2021

Functions mostly from: https://www.sfu.ca/~ssurjano/index.html

@author: Lisa Pretsch
"""
import numpy as np


def tosz(x):
    """
    Non-Linear Transformation
    Rn → Rn,  for any positive integer n (n = 1 and n = dim are used in the following), maps element-wise
    """
    xhat = np.where(x != 0, np.log(np.abs(x)), 0)
    c1 = np.where(x > 0, 7.9, 3.1)
    c2 = np.where(x > 0, 10.0, 5.5)

    return np.sign(x) * np.exp(xhat + 0.049 * (np.sin(c1 * xhat) + np.cos(c2 * xhat)))


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

        if self.coco_id == 1:  # F1: Sphere Function
            z = x - self.xopt
            f = np.sum(z ** 2, axis=-1) + self.fopt
            return f

        elif self.coco_id == 2:  # F2: Separable Ellipsoidal Function
            z = tosz(x - self.xopt)
            i = (np.arange(self.dim) + 1)
            f = np.sum(10 ** (6 * (i - 1) / (self.dim - 1)) * z ** 2, axis=-1) + self.fopt
            return f

        elif self.coco_id == 5:  # F5: Linear Slop
            z = np.where(self.xopt * x < 5 ** 2, x, self.xopt)
            i = (np.arange(self.dim) + 1)
            s = np.sign(self.xopt) * 10 ** ((i - 1) / (self.dim - 1))
            f = np.sum(5 * abs(s) - s * z, axis=-1) + self.fopt
            return f

    def grid(self, grid_size):
        """
        Generate an N-dimensional grid based on the parameter 'dim'
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
        grad_f = np.array(np.gradient(f.T))   # TODO: check if transpose works for higher dimensions
        grid_points = np.linspace(self.lb[1], self.ub[1], grid_size)

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
