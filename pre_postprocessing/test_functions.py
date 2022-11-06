#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wen Jun 30 14:51:33 2021

Functions mostly from: https://www.sfu.ca/~ssurjano/index.html

@author: Lisa Pretsch
"""
import numpy as np
from math import exp, pi


class TestFunction:
    # Instance attributes
    def __init__(self, name=None, dim=None, lb=None, ub=None):
        self.name = name        # name
        self.dim = dim          # dimensionality, number of design variables
        self.lb = lb            # lower bound of design variables as numpy array
        self.ub = ub            # upper bound of design variables as numpy array

    # Instance methods
    def __str__(self):
        return f'Test function: {self.name}, {self.dim}-dimensional, lower bound {self.lb}, upper bound {self.ub}.'
    
    def evaluate(self, x):
         print(f'Error: Method "evaluate" for {self.name} is not yet implemented.')


class Ackley(TestFunction):
# Usually evaluated on [-5,5]^d
# Global minimum at x_i=0 where f(x) = 0
# Many local optima
    def __init__(self, dim=2):
        super().__init__(
            name='Ackley',
            dim=dim,
            lb=-5*np.ones(dim),
            ub=5*np.ones(dim))
    
    def evaluate(self, x):
        a = 20
        b = 0.2
        c = 2*pi

        x = np.atleast_2d(x)
        sum1 = np.sum(x**2,axis=1)
        sum2 = np.sum(np.cos(c*x),axis=1)

        term1 = -a * np.exp(-b*np.sqrt(sum1/self.dim))
        term2 = -np.exp(sum2/self.dim)

        y = term1 + term2 + a + exp(1)
        return y
        # return -20*np.exp(-0.2*np.sqrt(0.5*(x[:,0]**2+x[:,1]**2))) - np.exp(0.5*(np.cos(2*np.pi*x[:,0])+np.cos(2*np.pi*x[:,1]))) + 20 + np.e


class Branin(TestFunction):
# Usually evaluated on x1=[-5,10], x2=[0,15]
# Three global minima at x=[-pi,12.275], [pi,2.275], [9.42478,2.475]
# where f(x)=0.397887
# Forrester 2008: +5*x[:,0] for two local & one global optimum
    def __init__(self):
        super().__init__(
            name='Branin',
            dim=2,
            lb=np.array([-5, 0]),
            ub=np.array([10, 15]))
    
    def evaluate(self, x):
        a = 1
        b = 5.1/(4*pi**2)
        c = 5/pi
        r = 6
        s = 10
        t = 1/(8*pi)

        x = np.atleast_2d(x)
        y = a*(x[:,1] - b*x[:,0]**2 + c*x[:,0]-r)**2 + s*(1-t)*np.cos(x[:,0]) + s
        return y


class Ishigami(TestFunction):
# Common example for uncertainty and sensitivity analysis methods,
# because it exhibits strong nonlinearity and nonmonotonicity
# Peculiar dependence on x3
# 
    def __init__(self):
        super().__init__(
            name='Ishigami',
            dim=3,
            lb=-pi*np.ones(3),
            ub=pi*np.ones(3))
    
    def evaluate(self, x):
        a = 7
        b = 0.1

        x = np.atleast_2d(x)
        y = np.sin(x[:,0]) + a*(np.sin(x[:,1]))**2 + b*(x[:,2])**4*np.sin(x[:,0])
        return y


class Quadratic(TestFunction):
# Can be evaluated on any search domain
# Global minimum at (0,0) where f(x1,x2)=0
# Simple quadratic function
    def __init__(self):
        super().__init__(
            name='Quadratic',
            dim=2,
            lb=-2*np.ones(2),
            ub=2*np.ones(2))
    
    def evaluate(self, x):
        x = np.atleast_2d(x)
        return x[:,0]**2 + x[:,1]**2


class Rosenbrock(TestFunction):
# Usually evaluated on [-2,2]^d
# Global minimum at x_i=1 where f(x)=0
    def __init__(self, dim=2):
        super().__init__(
            name='Rosenbrock',
            dim=dim,
            lb=-2*np.ones(dim),
            ub=2*np.ones(dim))
    
    def evaluate(self, x):
        a = 1
        b = 100

        x = np.atleast_2d(x)
        x_i = x[:,:-1]
        x_next = x[:,1:]
        
        term1 = b*(x_next-x_i**2)
        term2 = (x_i-a)**2

        y = np.sum(b*(x_next-x_i**2)**2 + (x_i-a)**2,axis=1)
        return y
        # return (a-x[:,0])**2 + b*(x[:,1] - x[:,0]**2)**2



class Styblinski(TestFunction):
# Usually evaluated on the hypercube [-5, 5]^d
# Global minimum at x_i=-2.903534 where f(x)=39.16599*d
    def __init__(self, dim=2):
        super().__init__(
            name='Styblinski-Tang',
            dim=dim,
            lb=-5*np.ones(dim),
            ub=5*np.ones(dim))
    
    def evaluate(self, x):
        x = np.atleast_2d(x)
        y = 0.5*(np.sum(x**4 - 16*x**2 + 5*x ,axis=1))
        return y



class Trid(TestFunction):
# Usually evaluated on the hypercube [-d^2, d^2]^d
# Global minimum at x_i=i*(d+1-i) where f(x)=-d*(d+4)(d-1)/6
# No local minimum except the global one
    def __init__(self, dim=2):
        super().__init__(
            name='Trid',
            dim=dim,
            lb=-dim**2*np.ones(dim),
            ub=dim**2*np.ones(dim))
    
    def evaluate(self, x):
        x = np.atleast_2d(x)
        x_i = x[:,1:]
        x_old = x[:,:-1]
        
        sum1 = np.sum((x-1)**2, axis=1)
        sum2 = np.sum(x_i*x_old, axis=1)

        y = sum1 - sum2
        return y



class Zakharov(TestFunction):
# Usually evaluated on the hypercube [-5, 10]^d
# Global minimum at x_i=0 where f(x)=0
# No local minimum except the global one
    def __init__(self, dim=2):
        super().__init__(
            name='Zakharov',
            dim=dim,
            lb=-5*np.ones(dim),
            ub=10*np.ones(dim))
    
    def evaluate(self, x):
        i = np.arange(self.dim)
        
        x = np.atleast_2d(x)
        
        sum1 = np.sum(x**2, axis=1)
        sum2 = np.sum(0.5*i*x, axis=1)

        y = sum1 + sum2**2 + sum2**4
        return y
