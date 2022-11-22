import numpy as np
import matplotlib.pyplot as plt


class Figure:
    def __init__(self):
        self.ax = None
        self.fig = None
    
    def contour(self, test_function):
        lb = 2 * test_function.lb
        ub = 2 * test_function.ub

        xlist = np.linspace(lb, ub, 100)
        ylist = np.linspace(lb, ub, 100)
        X, Y = np.meshgrid(xlist, ylist)
        inputs = np.array([X, Y])
        Z = test_function.evaluate(inputs)
        self.fig, self.ax=plt.subplots(1,1)
        
        cp = self.ax.contourf(X, Y, Z, cmap="rainbow")
        self.ax.contour(X, Y, Z, colors="black")
        self.fig.colorbar(cp) # Add a colorbar to a plot
        self.ax.set_title('Coco Function')
    
    def solution_point(self, solution):

        self.ax.plot(solution[0], solution[1], marker="o", markersize=10, markeredgecolor="red", markerfacecolor="white")
        plt.show()
