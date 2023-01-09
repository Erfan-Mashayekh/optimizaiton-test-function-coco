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
    dim = model['dimensions']  # input dimension ∈ {2,3,5,10,20,40}
    constrained = model['constrained']  # bool: constraints on/off
    num_constraints = model['num_constraints']  # number of constraints
    lb = model['lower_bounds'] * np.ones(dim)
    ub = model['upper_bounds'] * np.ones(dim)
    opt_read = model['opt_read']  # bool: read input points from data.csv
    grid_size = model['grid_size']  # grid size for plotting 

    # Optimum and initial points
    xopt = np.array([2, 1])
    fopt = 10
    ystar = np.array([-4, 2])
    # xinit = np.random.uniform(low=lb, high=ub, size=(dim))
    xinit = lb + (ub-lb)/2
    x = [xopt, fopt, ystar]

    # Test function
    test_function = COCOTestFunction(model, dim, constrained, num_constraints, lb, ub, xopt, fopt, ystar)

    # Get responses: objective function f and constraints g
    def get_f(x):
        return test_function.evaluate(x)

    def get_g(x, a):
        return test_function.compute_g(x, a)

    def get_resp(x, a):
        f = test_function.evaluate(x)
        f = f.reshape(1, -1)
        g = test_function.compute_g(x, a)
        return np.concatenate((f, g))

    # Optimization process
    time_start = timer()
    if function_type == 1:  # 1: coco function
        if constrained:
            a = test_function.initialize_constraint_parameters(grid_size)
            cons = {'type': 'ineq', 'fun': lambda x: get_g(x, a)}
        else:
            a = None
            cons = ()
        bounds = Bounds(lb, ub)
        opti_result = minimize(lambda x: get_f(x),
                                xinit, args=(),
                                method='COBYLA',
                                constraints=cons,
                                bounds=bounds,
                                tol=1e-3)

    else:  # other types of functions
        pass # TODO

    # Print & plot results
    time_opti = timer() - time_start
    print(f"The solution is :\n {opti_result}")
    print("Optimization finished after %f seconds." % time_opti)
    if dim == 2:
        figure = Figure(dim, model)
        figure.contour(test_function, grid_size, a)
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
        g_test = get_g(x_test,a)
        resp_test = get_resp(x_test,a)
        if g_test.shape != (num_constraints, s):
            warn('Constraint functions can not handle vectorized inputs.')
        if resp_test.shape != (num_constraints+1, s):
            warn('Response functions can not handle vectorized inputs.')

# %%
