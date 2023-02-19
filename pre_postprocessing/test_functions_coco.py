# %%
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wen Jun 30 14:51:33 2021

Functions mostly from: https://www.sfu.ca/~ssurjano/index.html

@author: Lisa Pretsch, Erfan Mashayekh
"""
import numpy as np
from typing import Dict, Union, List, Tuple

from ioh import problem
from ioh import get_problem, ProblemType


class COCOTestFunction:

    def __init__(self, model: Dict[str, Union[int, str, bool, List[float]]], dim: int, instance: int,
                constrained: bool, num_constraints: int, lb: np.ndarray, ub: np.ndarray, ystar: np.ndarray) -> None:
        """
        Initialize the problem instance.

        Args:
        - model (Dict[str, Union[int, str, bool, List[float]]]): a dictionary containing model information, including the test function name
        - dim (int): the dimension of the problem
        - instance (int): the instance of the problem
        - constrained (bool): a flag indicating whether the problem is constrained or not
        - num_constraints (int): the number of constraints of the problem
        - lb (np.ndarray): the lower bound of the search space
        - ub (np.ndarray): the upper bound of the search space
        - ystar (float): the target function value
        """
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

    def evaluate(self, x: np.ndarray) -> Union[float, np.ndarray]:
        """
        Compute the value of the function for the corresponding input 'x'

        Args:
        - x (np.ndarray): the input to the function

        Returns:
        - Union[float, np.ndarray]: the output of the function
        """
        if x.size <= self.dim:
            # If the size of x is less than or equal to the dimension of the function, evaluate the function at x
            return self.problem(x)
        else:
            # Otherwise, evaluate the function at each row of x and return the resulting array
            function_list = np.empty(x.shape[0])
            for i in range(x.shape[0]):
                function_list[i] = self.problem(x[i])
            return function_list

    def grid(self, grid_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a d-dimensional grid based on the parameter 'dim'

        Args:
        - grid_size (int): the size of the grid

        Returns:
        - Tuple[np.ndarray, np.ndarray]: a tuple of two NumPy arrays, the first being the grid points, and the second being the evaluated function values at those grid points
        """
        args = []
        for i in range(self.dim):
            args.append(np.linspace(self.lb[0], self.ub[0], grid_size))        
        mesh = np.array(np.meshgrid(*args))
        mesh_reshaped = mesh.reshape((mesh.shape[0], mesh[0].size)).T
        f = self.evaluate(mesh_reshaped)

        return mesh, f

    def compute_a(self, alpha: float, grid_size: int, constraints_seed: int) -> None:
        """
        Compute coefficient 'a' which helps to generate a random slope for a linear constraint.

        Args:
        - alpha (float): the coefficient alpha used to compute the value of 'a'
        - grid_size (int): the size of the grid used to compute the gradient
        - constraints_seed (int): the seed used for random generation of 'a'

        Returns:
        - None
        """
        # Generate 3 grid points around ystar in each coordinate and 
        # compute the gradient at this point using 2nd order finite difference method
        delta = (self.ub[0]-self.lb[0]) / (grid_size-1)
        print(f'delta : {delta}')
        grad_f_ystar = np.empty(self.dim)
        for i in range(self.dim):
            lower_point = np.copy(self.ystar)
            lower_point[i] = lower_point[i] - delta            
            upper_point = np.copy(self.ystar)
            upper_point[i] = self.ystar[i] + delta
            grad_f_ystar[i] = (self.problem(upper_point) - self.problem(lower_point)) / (2*delta)

        # compute the value of coefficient 'a' based on ystar position and its gradient
        self.a = np.zeros((self.num_constraints, self.dim))
        self.a[0, :] = - alpha * grad_f_ystar / np.linalg.norm(grad_f_ystar)
        np.random.seed(constraints_seed)
        self.a[1:, :] = np.random.normal(self.a[0, :], 0.9, size=(self.num_constraints - 1, self.dim))


    def initialize_constraint_parameters(self, grid_size: int, constraints_seed: int) -> np.ndarray:
        """
        Initialize the constraint parameters 'a' and 'b' using the given grid size and constraints seed.

        Args:
        - grid_size (int): the size of the grid used to compute the gradient
        - constraints_seed (int): the seed used for random generation of 'a'

        Returns:
        - np.ndarray: the constraint parameter 'a'
        """
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

    def optimization_error(self, result, ystar):
        """
        Computes the error of the optimal point
        """
        # Calculate the difference between the optimized function value and the actual optimal value
        # If the problem is not constrained, calculate the difference between the optimized function value and 
        # the actual optimal value of the problem.
        if self.constrained:
            #TODO: ystar is not the optimal point, hence this computation is not correct. A proper way need to be considered.
            return result['fun'] - self.evaluate(ystar) 
        else:
            return result['fun'] - self.problem.optimum.y

# %%
