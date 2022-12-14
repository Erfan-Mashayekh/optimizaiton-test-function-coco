import numpy as np
import matplotlib.pyplot as plt

from optimizer import Optimizer
from timeit import default_timer as timer

from utils import read_model
from pre_postprocessing.figure import Figure

if __name__ == '__main__':

    model = read_model()
    function_type = model['function_type']  # types of functions. coco:1, normal:2
    opt_read = model['opt_read']  # bool: read input points from data.csv
    dim = model['dimensions']  # input dimension ∈ {2,3,5,10,20,40}
    grid_size = model['grid_size']  # grid size for plotting

    xopt = np.array([2, 1])
    fopt = 10
    ystar = np.array([-4, 2])
    #x_init = np.array([[4, 3], [-4, 3], [-4, -3]])
    x_init = np.random.uniform(low=[-5, -5], high=[5, 5], size=(model['points'], dim))
    x = [xopt, fopt, ystar]

    optimizer = []
    result = []

    for x_init_elm in x_init:
        optimizer.append(Optimizer(model, x, x_init_elm))
    time_start = timer()
    if function_type == 1:
        counter = 0
        cons = {}
        for opt in optimizer:
            res, test_function, cons = opt.run_coco(counter, cons)
            result.append(res)
            counter += 1
    else:
        pass
    time_opti = timer() - time_start
    print(f"The solution is :\n {result}")
    print("Optimization finished after %f seconds." % time_opti)

    if dim == 2:
        figure = Figure(dim)
        a = optimizer[0].get_a()
        figure.contour(test_function, grid_size, a)
        figure.init_points(model['constrained'], x_init, ystar)
        for res in result:
            figure.solution_points(res["x"])
        plt.show()
