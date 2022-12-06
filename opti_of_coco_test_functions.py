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
from functools import partial

from pre_postprocessing.coco_test_functions import COCOTestFunction
from pre_postprocessing.figure import Figure

# COCO TEST FUNCTION
coco_id = 1
constrained = True             # bool
dim = 2                         # input dimension ∈ {2,3,5,10,20,40}
# xopt = np.ones((dim,))        # location of optimal point
xopt = np.array([-5,5])
fopt = 10                       # optimal value        
ystar = np.array([-4,2])        # location of constrained optimal point
constr_number = 5

test_function = COCOTestFunction(id=coco_id, constrained=constrained, constr_number=constr_number, dim=dim, xopt=xopt, fopt=fopt, ystar=ystar) # optimization problem definition
name = test_function.name         # name of test function
lb = test_function.lb             # array of design variable lower bounds (np.array (1,n) or None)
ub = test_function.ub             # array of design variable upper bounds (np.array (1,n) or None)
x_init = 10 * np.random.random((dim,)) # array of initial design variable values (np.array (1,n) or None)
x_init = np.array([6,6])

def object_function(x):
    return test_function.evaluate(x)

# OPTIMIZER
method = 'COBYLA'              # BFGS, COBYLA, SLSQP, ...
bounds = Bounds(lb=lb, ub=ub)  # only for Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell, and trust-constr methods
if constrained:
    test_function.initialize_constraint_parameters()
    cons = {'type': 'ineq', 'fun' : lambda x: test_function.compute_g(x)}


time_start = timer()

if constrained:
    opti_result = minimize(object_function, 
                        x_init, args=(), 
                        method=method,
                        constraints=cons,
                        bounds=bounds,
                        tol=1e-3)
else:
    opti_result = minimize(object_function, 
                    x_init, args=(), 
                    method=method,
                    bounds=bounds,
                    tol=1e-3)    

# Print results and simulation time
print(f"The solution is :\n {opti_result}")
time_opti = timer()-time_start
print("Optimization finished after %f seconds." % time_opti)

# Plot convergence
# TODO: plot convergence of optimization objective (& constraints), i.e., of all response variables
if (dim == 2):
    figure = Figure(dim)
    figure.contour(test_function)
    figure.solution_point(constrained, x_init, opti_result["x"], ystar)