import numpy as np
import matplotlib.pyplot as plt

from optimizer import Optimizer
from timeit import default_timer as timer

from utils import read_model
from pre_postprocessing.figure import Figure

if __name__ == '__main__':

    model = read_model()
    scenario = model['scenario']
    function_type = model['function_type']  # types of functions. coco:1, normal:2
    opt_read = model['opt_read']  # bool: read input points from data.csv
    dim = model['dimensions']  # input dimension ∈ {2,3,5,10,20,40}
    grid_size = model['grid_size']  # grid size for plotting
    lb = model['lower_bounds'] * np.ones(dim)
    ub = model['upper_bounds'] * np.ones(dim)

    xopt = np.array([2, 1])
    fopt = 10
    ystar = np.array([-4, 2])
    points = model['points']
    x_init = np.random.uniform(low=lb, high=ub, size=(points, dim))
    x = [xopt, fopt, ystar]

    if scenario == 'evaluate':  # evaluate objective function and constraint computation
        optimizer = (Optimizer(model, x, x_init))
        counter = 0
        cons = {}
        f = optimizer.get_f(x_init)
        g = optimizer.get_g(x_init).T
        print(f'function (points, dimension): {f.shape} \n {f}')
        print(f'constraints (points, constraints number): {g.shape} \n {g}')
        exit()
    elif scenario == 'optimize':  # optimize the points
        optimizer = []
        result = []
        for x_init_elm in x_init:
            optimizer.append(Optimizer(model, x, x_init_elm))
        time_start = timer()
        if function_type == 1:  # 1: coco function
            counter = 0
            cons = {}
            for opt in optimizer:
                res, test_function, cons = opt.run_coco(counter, cons)
                result.append(res)
                counter += 1
        else:  # other types of functions
            pass

        time_opti = timer() - time_start
        print(f"The solution is :\n {result}")
        print("Optimization finished after %f seconds." % time_opti)
        if dim == 2:
            figure = Figure(dim, model)
            a = optimizer[0].get_a()
            figure.contour(test_function, grid_size, a)
            figure.init_points(model['constrained'], x_init, ystar)
            for res in result:
                figure.solution_points(res["x"])
            plt.show()
