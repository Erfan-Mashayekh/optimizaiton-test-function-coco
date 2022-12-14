# %%
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 11:46 2022

@author: Lisa Pretsch
"""
import numpy as np
from timeit import default_timer as timer
# Optimization
from scipy.optimize import minimize, Bounds
# Own modules
from pre_postprocessing.test_functions import Ackley, Rosenbrock, Styblinski, Trid, Zakharov

# MY TEST FUNCTION
dim = 20
test_function = Styblinski(dim)  # objective function (Ackley, Rosenbrock, Styblinski, Trid, Zakharov)
lb = np.min(test_function.lb)
ub = np.max(test_function.ub)
x0 = np.zeros((dim,))


def response(x):
    x = np.asarray(x)
    x = x.reshape(1, -1)
    return test_function.evaluate(x)


# OPTIMIZER
method = 'SLSQP'  # BFGS, COBYLA, SLSQP, ...
bounds = Bounds(lb=lb, ub=ub)  # only for Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell, and trust-constr methods

time_start = timer()
opti_result = minimize(response, x0, args=(), method=method, bounds=bounds)
print(opti_result)
time_opti = timer() - time_start
print("Optimization finished after %f seconds." % time_opti)
