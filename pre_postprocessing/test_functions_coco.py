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

    def __init__(self, model, dim, instance, constrained, num_constraints, lb, ub, ystar):

        self.coco_id = model['coco_id']  # name of test function
        self.dim = dim
        self.instance = instance
        self.constrained = constrained
        self.num_constraints = num_constraints
        self.lb = lb
        self.ub = ub
        self.ystar = ystar
        self.problem = get_problem(self.coco_id, self.instance, self.dim, ProblemType.BBOB)  
        self.a = np.zeros((self.num_constraints, self.dim))
        

    def evaluate(self, x):
        """
        Compute the value of the function for the corresponding input 'x'
        """
        if x.size <= self.dim:
            return self.problem(x)
        else:
            function_list = np.empty(x.shape[0])
            for i in range(x.shape[0]):
                function_list[i] = self.problem(x[i]) 
            return function_list


    def grid(self, grid_size):
        """
        Generate an d-dimensional grid based on the parameter 'dim'
        :param grid_size: the size of the grid
        :return: x:
        """
        args = []
        for i in range(self.dim):
            args.append(np.linspace(self.lb[0], self.ub[0], grid_size))        
        mesh = np.array(np.meshgrid(*args))
        mesh_reshaped = mesh.reshape((mesh.shape[0], mesh[0].size)).T
        f = self.evaluate(mesh_reshaped)

        return mesh, f

    def compute_a(self, alpha, grid_size, constraints_seed):
        """
        Compute coefficient 'a' which helps to generate a random slope for a linear constraint.
        """
        # Generate 3 grid points around ystar in each coordinate
        args = []
        delta = (self.ub[0]-self.lb[0]) / (grid_size-1)
        for i in range(self.dim):
            grid_points = np.array([self.ystar[i] - delta, 
                                    self.ystar[i],
                                    self.ystar[i] + delta])
            args.append(grid_points)  

        mesh = np.array(np.meshgrid(*args))
        mesh_reshaped = mesh.reshape((mesh.shape[0], mesh[0].size)).T

        # evaluate the coco function and the its gradients in this small region
        f = self.evaluate(mesh_reshaped)
        f = f.reshape(mesh[0].shape)
        grad_f = np.array(np.gradient(f.T))

        # allocate the gardient of the center point to ystar at each coordinate
        dim = 0
        grad_f_ystar = np.empty(self.dim)
        for grad_dim in grad_f:
            index = np.ones(self.dim, dtype=np.uint32)
            grad_f_ystar[dim] = grad_dim[tuple(index)]
            dim =+ 1

        # compute the value of coefficient 'a' based on ystar poisition and its gradiend
        self.a = np.zeros((self.num_constraints, self.dim))
        self.a[0, :] = - alpha * grad_f_ystar / np.linalg.norm(grad_f_ystar)
        np.random.seed(constraints_seed)
        self.a[1:, :] = np.random.normal(self.a[0, :], 0.9, size=(self.num_constraints - 1, self.dim))

    def initialize_constraint_parameters(self, grid_size, constraints_seed):
        alpha = np.ones(self.num_constraints)
        b = np.zeros(self.num_constraints)
        self.compute_a(alpha[0], grid_size, constraints_seed)
        return self.a

    def compute_g(self, x, a):
        """
        Compute the linear constraints in the "d-dimensional" space
        """
        # g = alpha * (a @ (x-self.ystar).T) + b
        g = (a @ (x - self.ystar).T)
        g[1] = g[1] + 2.
        return -g

    def optimization_error(self, result):
        """
        Computes the error of the optimal point
        """
        return np.linalg.norm(result['x'] - self.problem.optimum.x)