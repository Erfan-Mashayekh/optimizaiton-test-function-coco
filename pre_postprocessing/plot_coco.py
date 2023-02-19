import matplotlib.pyplot as plt
import numpy as np


class Figure:
    def __init__(self, dim, model):
        self.log_scale = model['log_scale']
        self.ax = None
        self.fig = None
        self.dim = dim

    def contour(self, test_function, grid_size, a, b):
        mesh, f = test_function.grid(grid_size)
        f = f.reshape(mesh[0].shape)

        self.fig, self.ax = plt.subplots(1, 1)
        if self.log_scale:
            cp = self.ax.contour(mesh[0], mesh[1], np.log(f), cmap='rainbow')
        else:
            cp = self.ax.contour(mesh[0], mesh[1], f, cmap='rainbow')
        self.fig.colorbar(cp)

        x = np.zeros((f.size, self.dim))
        x[:, 0] = np.reshape(mesh[0], f.size)
        x[:, 1] = np.reshape(mesh[1], f.size)

        if test_function.constrained:
            alpha = 0.9
            b = b.reshape((test_function.num_constraints,1))
            g = test_function.compute_g(x, a, b)
            for i in range(test_function.num_constraints):
                g_reshaped = np.reshape(g[i, :], f.shape) < 0
                self.ax.contourf(mesh[0], mesh[1], g_reshaped, cmap='Greys', alpha=alpha)
                self.ax.contour(mesh[0], mesh[1], g_reshaped,
                                linewidths=0.1,
                                colors='black')
                alpha = alpha * 0.7
        self.ax.set_title('Coco Function')

    def init_points(self, constrained, xinit, ystar):
        self.ax.plot(xinit[0], xinit[1], 
                        marker="o", markersize=4,
                        markeredgecolor="black", markerfacecolor="blue")
        if constrained:
            self.ax.plot(ystar[0], ystar[1],
                         marker="*", markersize=15,
                         markeredgecolor="black", markerfacecolor="yellow")

    def solution_points(self, solution):
        self.ax.plot(solution[0], solution[1],
                     marker="o", markersize=4,
                     markeredgecolor="black", markerfacecolor="red")
