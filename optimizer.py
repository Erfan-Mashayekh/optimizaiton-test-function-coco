# %%
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 11:46 2022

@author: Lisa Pretsch
"""
import numpy as np
from scipy.optimize import minimize, Bounds

from pre_postprocessing.coco_test_functions import COCOTestFunction


class Optimizer:

    def __init__(self, model, x, x_init):
        self.coco_id = model['coco_id']  # name of test function
        self.constrained = model['constrained']  # bool: constraints on/off
        self.dim = model['dimensions']  # input dimension ∈ {2,3,5,10,20,40}
        self.points = model['points']  # number of points to be minimized
        self.method = model['min_method']  # minimization method: BFGS, COBYLA, SLSQP, ...
        self.constr_number = model['constraints']  # number of constraints
        self.lb_mag = model['lower_bounds']  # lower bounds magnitude
        self.ub_mag = model['upper_bounds']  # upper bounds magnitude
        self.grid_size = model['grid_size']

        self.xopt = x[0]
        self.fopt = x[1]
        self.ystar = x[2]
        self.x_init = x_init

    def run_coco(self, counter, cons):

        lb = self.lb_mag * np.ones(self.dim)  # array of design variable lower bounds
        ub = self.ub_mag * np.ones(self.dim)  # array of design variable upper bounds
        bounds = Bounds(lb=lb, ub=ub)
        test_function = COCOTestFunction(self.coco_id,
                                         self.dim, lb, ub, self.points,
                                         self.constrained, self.constr_number,
                                         self.xopt, self.fopt, self.ystar)
        if self.constrained and counter == 0:
            self.a = test_function.initialize_constraint_parameters(self.grid_size)
            cons = {'type': 'ineq', 'fun': lambda x: test_function.compute_g(x, self.a)}
        if self.constrained:
            opti_result = minimize(lambda x: test_function.evaluate(x),
                                   self.x_init, args=(),
                                   method=self.method,
                                   constraints=cons,
                                   bounds=bounds,
                                   tol=1e-3)
        else:
            opti_result = minimize(lambda x: test_function.evaluate(x),
                                   self.x_init, args=(),
                                   method=self.method,
                                   bounds=bounds,
                                   tol=1e-3)

        return opti_result, test_function, cons

    def get_a(self):
        return self.a
