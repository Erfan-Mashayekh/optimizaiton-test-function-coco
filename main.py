# %%
import json
import numpy as np
import matplotlib.pyplot as plt
from warnings import warn

from scipy.optimize import minimize, Bounds
from timeit import default_timer as timer

from pre_postprocessing.test_functions_coco import COCOTestFunction
from pre_postprocessing.plot_coco import Figure

if __name__ == '__main__':

    # Settings from model.json
    with open('model.json', 'r') as handle:
        model = json.load(handle)
    function_type = model['function_type']  # types of functions. coco:1, normal:2
    coco_id = model['coco_id'] # COCO test function id
    instance = model['instance'] # instance id of the IOH problem
    dim = model['dimensions']  # input dimension ∈ {2,3,5,10,20,40}
    constrained = model['constrained']  # bool: constraints on/off
    num_constraints = model['num_constraints']  # number of constraints
    seed = model['seed']  # specified seed for random constraints and inputs
    lb = model['lower_bounds'] * np.ones(dim)
    ub = model['upper_bounds'] * np.ones(dim)
    opt_read = model['opt_read']  # bool: read input points from data.csv
    grid_size = model['grid_size']  # grid size for plotting 

    # Optimum and initial points
    np.random.seed(seed)
    ystar = np.random.uniform(low=lb, high=ub, size=(dim))
    xinit = np.random.uniform(low=lb, high=ub, size=(dim))

    # Test function
    test_function = COCOTestFunction(model, dim, instance, constrained, num_constraints, lb, ub, ystar)

    # Get responses: objective function f and constraints g
    def get_f(x):
        return test_function.evaluate(x)

    def get_g(x, a, b):
        return test_function.compute_g(x, a, b)

    def get_resp(x, a, b):
        f = test_function.evaluate(x)
        f = f.reshape(1, -1)
        g = test_function.compute_g(x, a, b)
        return np.concatenate((f, g))

    # Optimization process
    time_start = timer()
    if function_type == 1:  # 1: coco function
        if constrained:
            a, b = test_function.initialize_constraint_parameters(grid_size, seed)
            cons = {'type': 'ineq', 'fun': lambda x: get_g(x, a, b)}
        else:
            a = None
            b = None
            cons = ()
        bounds = Bounds(lb, ub)
        opti_result = minimize(lambda x: get_f(x),
                                xinit, args=(),
                                method='SLSQP',
                                constraints=cons,
                                bounds=bounds,
                                tol=1e-3)
    else:  # other types of functions
        pass # TODO

    # Print results
    time_opti = timer() - time_start
    print(f"Optimization time: {time_opti} seconds.")
    print(f"Optimization solution:\n {opti_result}")
    print(f"Optimization error: {test_function.optimization_error(opti_result, ystar)}")      
        
    # Plot results
    if dim == 2:
        figure = Figure(dim, model)
        figure.contour(test_function, grid_size, a, b)
        figure.init_points(constrained, xinit, ystar)
        figure.solution_points(opti_result["x"])
    plt.show()

    # Test vectorization ability
    """
    The objective function is sent an x array with x.shape == (s, dim),
    and is expected to return an array of shape (s,),
    where s is the number of solution vectors to be calculated.
    If constraints are applied, the constraint function is sent an 
    x array with x.shape == (s, dim), and is expected to return an array
    of shape (m, s), where m is the number of constraint components.
    """
    s=5
    x_test = np.random.uniform(low=lb, high=ub, size=(s,dim))
    
    f_test = get_f(x_test)
    if f_test.shape != (s,):        
        warn('Objective function can not handle vectorized inputs.')
    
    if constrained:        
        g_test = get_g(x_test,a, b.reshape((num_constraints,1)))
        resp_test = get_resp(x_test,a, b.reshape((num_constraints,1)))
        if g_test.shape != (num_constraints, s):
            warn('Constraint functions can not handle vectorized inputs.')
        if resp_test.shape != (num_constraints+1, s):
            warn('Response functions can not handle vectorized inputs.')

# %%