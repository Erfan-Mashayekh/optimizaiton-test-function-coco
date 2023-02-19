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
        self.b = np.zeros(self.num_constraints)

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

    def random_sample(self, n) -> np.ndarray:
        """
        Generate a random sample from an isotropic multivariate normal distribution

        Returns:
            np.ndarray: A random sample of size (n,).
        """
        # generate the vector of standard normal random variables
        z = np.random.normal(size=n)

        # generate the matrix of eigenvectors
        eigenvec = np.linalg.qr(np.random.normal(size=(n, n)))[0]

        # scale the eigenvectors by the square root of the eigenvalues
        eigenvec = eigenvec / np.sqrt(n)

        # compute the sample
        sample = eigenvec @ z

        return sample

    def compute_a(self, alpha: float, grid_size: int, constraints_seed: int) -> None:
        """
        Compute coefficient 'a' as a random slope for a linear constraint.

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

        # compute the value of coefficient 'a1' based on ystar position and its gradient
        self.a = np.zeros((self.num_constraints, self.dim))
        self.a[0, :] = - alpha * grad_f_ystar / np.linalg.norm(grad_f_ystar)
        
        # compute a_k by randomly sampling from an isotropic multivariate normal distribution
        for k in range(self.num_constraints-1):
            self.a[k+1, :] = self.random_sample(self.dim)

        # alternative way with seed
        # np.random.seed(seed)
        # self.a[1:, :] = np.random.normal(self.a[0, :], 0.9, size=(self.num_constraints - 1, self.dim))

    def compute_b(self) -> None:
        """
        Compute coefficient 'a' as a random bias for a linear constraint.

        Returns:
        - None
        """
        self.b[0] = 0
        
        # compute b_k by randomly sampling from an isotropic multivariate normal distribution
        self.b[1:] = self.random_sample(self.num_constraints-1)
        # TODO: A constraint is inactive at the constructed optimum if bk > 0
        self.b = np.where(self.b <= 0, self.b, -self.b) 

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
        self.compute_a(alpha[0], grid_size, constraints_seed)
        self.compute_b()
        return self.a, self.b

    def compute_g(self, x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute the value of the linear constraints in the "d-dimensional" space using the formula g = (a @ (x - ystar).T) + b.
        Args:
        - x (numpy.ndarray): an array containing the input points, where each row represents a single point and each column represents a dimension.
        - a (numpy.ndarray): an array containing the normal vectors of the linear constraints.
        - b (numpy.ndarray): an array containing the intercepts of the linear constraints.
        
        Returns:
        - g (numpy.ndarray): an array containing the values of the linear constraints at the given input points.
        """
        g = (a @ (x - self.ystar).T) + b
        return -g

    def optimization_error(self, result, ystar):
        """
        Compute the error of the optimal point
        """
        # Calculate the difference between the optimized function value and the actual optimal value
        # If the problem is not constrained, calculate the difference between the optimized function value and 
        # the actual optimal value of the problem.
        if self.constrained:
            print("Warning: Note that ystar is a unique optimal point if f(x) is strictly pseudoconvex at ystar")
            return result['fun'] - self.evaluate(ystar) 
        else:
            return result['fun'] - self.problem.optimum.y

# %%
