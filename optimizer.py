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
        self.model = model
        self.scenario = model['scenario']
        self.constrained = model['constrained']  # bool: constraints on/off
        self.dim = model['dimensions']  # input dimension ∈ {2,3,5,10,20,40}
        self.method = model['min_method']  # minimization method: BFGS, COBYLA, SLSQP, ...
        self.lb_mag = model['lower_bounds']  # lower bounds magnitude
        self.ub_mag = model['upper_bounds']  # upper bounds magnitude
        self.grid_size = model['grid_size']

        self.xopt = x[0]
        self.fopt = x[1]
        self.ystar = x[2]
        self.x_init = x_init

        self.lb = self.lb_mag * np.ones(self.dim)  # array of design variable lower bounds
        self.ub = self.ub_mag * np.ones(self.dim)  # array of design variable upper bounds
        self.test_function = COCOTestFunction(self.model, self.dim, self.lb, self.ub, self.constrained,
                                              self.xopt, self.fopt, self.ystar)
        self.a = None

    def get_f(self, x):
        return self.test_function.evaluate(x)

    def get_g(self, x):
        if self.scenario == 'evaluate':
            self.a = self.test_function.initialize_constraint_parameters(self.grid_size)
            return self.test_function.compute_g(x, self.a)
        elif self.scenario == 'optimize':
            return self.test_function.compute_g(x, self.a)

    def run_coco(self, counter, cons):
        bounds = Bounds(lb=self.lb, ub=self.ub)
        if self.constrained and counter == 0:
            self.a = self.test_function.initialize_constraint_parameters(self.grid_size)
            cons = {'type': 'ineq', 'fun': lambda x: self.get_g(x)}
        if self.constrained:
            opti_result = minimize(lambda x: self.get_f(x),
                                   self.x_init, args=(),
                                   method=self.method,
                                   constraints=cons,
                                   bounds=bounds,
                                   tol=1e-3)
        else:
            opti_result = minimize(lambda x: self.get_f(x),
                                   self.x_init, args=(),
                                   method=self.method,
                                   bounds=bounds,
                                   tol=1e-3)

        return opti_result, self.test_function, cons

    def get_a(self):
        return self.a
