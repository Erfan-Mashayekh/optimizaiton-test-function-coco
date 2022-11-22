#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 11:46 2022

@author: Lisa Pretsch
"""
import numpy as np
from timeit import default_timer as timer
from scipy.optimize import minimize, Bounds

from pre_postprocessing.coco_test_functions import COCOTestFunction
from pre_postprocessing.figure import Figure

# COCO TEST FUNCTION
coco_id = 2
constrained = True      # bool
dim = 2                 # input dimension ∈ {2,3,5,10,20,40}
xopt = np.ones((dim,))  # location of optimal x
fopt = 10               # optimal value        

test_function = COCOTestFunction(id=coco_id, constr=constrained, dim=dim, xopt=xopt, fopt=fopt) # optimization problem definition
name = test_function.name         # name of test function
lb = test_function.lb             # array of design variable lower bounds (np.array (1,n) or None)
ub = test_function.ub             # array of design variable upper bounds (np.array (1,n) or None)
x_init = np.random.random((dim,)) # array of initial design variable values (np.array (1,n) or None)

def response(x):
    return test_function.evaluate(x)
    
# TODO: create coco_test_function.py with class COCOTestFunction

# OPTIMIZER
method = 'SLSQP'              # BFGS, COBYLA, SLSQP, ...
bounds = Bounds(lb=lb, ub=ub) # only for Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell, and trust-constr methods

time_start = timer()
opti_result = minimize(response, x_init, args=(), method=method, bounds=bounds)

# Print results and simulation time
print(f"The solution is :\n {opti_result}")
time_opti = timer()-time_start
print("Optimization finished after %f seconds." % time_opti)

# Plot convergence
# TODO: plot convergence of optimization objective (& constraints), i.e., of all response variables
if (dim == 2):
    figure = Figure()
    figure.contour(test_function)
    figure.solution_point(opti_result["x"])