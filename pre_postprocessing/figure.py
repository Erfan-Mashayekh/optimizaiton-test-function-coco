import matplotlib.pyplot as plt
import numpy as np


class Figure:
    def __init__(self, dim):
        self.ax = None
        self.fig = None
        self.dim = dim

    def contour(self, test_function, grid_size, a):
        x1_mesh, x2_mesh, f = test_function.grid(grid_size)
        f = f.reshape(x1_mesh.shape)

        self.fig, self.ax = plt.subplots(1, 1)
        cp = self.ax.contour(x1_mesh, x2_mesh, f, cmap='rainbow')
        self.fig.colorbar(cp)

        x = np.zeros((f.size, self.dim))
        x[:, 0] = np.reshape(x1_mesh, f.size)
        x[:, 1] = np.reshape(x2_mesh, f.size)
        # print(f'x: \n{x}')
        if test_function.constrained:
            alpha = 0.9
            #a = test_function.get_a()
            print(a)
            g = test_function.compute_g(x, a)
            for i in range(test_function.constr_number):
                g_reshaped = np.reshape(g[i, :], f.shape) < 0
                #print(f' \n {g_reshaped}')
                self.ax.contourf(x1_mesh, x2_mesh, g_reshaped,
                                 cmap='Greys',
                                 alpha=alpha)
                self.ax.contour(x1_mesh, x2_mesh, g_reshaped,
                                linewidths=1.0,
                                levels=0,
                                colors='black')
                alpha = alpha * 0.7

        self.ax.set_title('Coco Function')

    def init_points(self, constrained, init, ystar):
        self.ax.scatter(init[:, 0], init[:, 1], s=50, c='blue')
        if constrained:
            self.ax.plot(ystar[0], ystar[1],
                         marker="*", markersize=7,
                         markeredgecolor="black", markerfacecolor="yellow")

    def solution_points(self, solution):
        self.ax.plot(solution[0], solution[1],
                     marker="o", markersize=7,
                     markeredgecolor="black", markerfacecolor="red")
