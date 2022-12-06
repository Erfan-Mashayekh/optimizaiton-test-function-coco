
import matplotlib.pyplot as plt
import numpy as np

class Figure:
    def __init__(self, dim):
        self.ax = None
        self.fig = None
        self.dim = dim
    
    def contour(self, test_function):
        x1_mesh, x2_mesh, f = test_function.grid()
        self.fig, self.ax=plt.subplots(1,1)    
        cp = self.ax.contour(x1_mesh, x2_mesh, (f), cmap='rainbow')        
        self.fig.colorbar(cp)

        x = np.zeros((f.size, self.dim))
        x[:, 0] = np.reshape(x1_mesh, f.size)
        x[:, 1] = np.reshape(x2_mesh, f.size)
        
        if test_function.constrained:
            alpha = 0.9
            g = test_function.compute_g(x)
            for i in range(test_function.constr_number):
                g_reshaped = np.reshape(g[i,:], f.shape) < 0         
                self.ax.contourf(x1_mesh, x2_mesh, g_reshaped, 
                                cmap='Greys',
                                alpha=alpha)
                self.ax.contour(x1_mesh, x2_mesh, g_reshaped, 
                                linewidths=1.0,
                                levels=0,
                                colors='black')        
                alpha = alpha * 0.7

        self.ax.set_title('Coco Function')
    
    def solution_point(self, constrained, init, solution, ystar):
        self.ax.plot(solution[0], solution[1], marker="o", markersize=8, markeredgecolor="black", markerfacecolor="red")
        self.ax.plot(init[0], init[1], marker="o", markersize=8, markeredgecolor="black", markerfacecolor="blue")
        if constrained:
            self.ax.plot(ystar[0], ystar[1], marker="*", markersize=7, markeredgecolor="black", markerfacecolor="yellow")
        plt.show()