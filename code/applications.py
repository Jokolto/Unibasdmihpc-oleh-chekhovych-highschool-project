import numpy as np
import matplotlib.pyplot as plt
import time
import csv
from numba import njit, int32

from numba.openmp import openmp_context as omp
from numba.openmp import omp_set_num_threads, omp_get_thread_num


class BaseLineMandelbrot:
    def __init__(self, xmin=-2, xmax=1, ymin=-1.5, ymax=1.5, num_points=2000, max_iterations=500, bound=2):
        self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
        self.num_points, self.max_iterations, self.bound = num_points, max_iterations, bound
        self.xDomain = np.linspace(xmin, xmax, num_points)
        self.yDomain = np.linspace(ymin, ymax, num_points)
        self.iterationArray = np.zeros((num_points, num_points), dtype=int)
    
    
    def mandelbrot_calculate(self, z, p, c):
        return z**p + c

    
    def compute_pixel(self, c):
        z = 0
        p = 2
        for iteration in range(self.max_iterations):
            if abs(z) >= self.bound:
                return iteration
            z = self.mandelbrot_calculate(z, p, c)
            
        return self.max_iterations
    
    def compute(self):
        for y_i in range(len(self.yDomain)):
            for x_i in range(len(self.xDomain)):
                c = complex(self.xDomain[x_i], self.yDomain[y_i])
                self.iterationArray[y_i, x_i] = self.compute_pixel(c)
        return self.iterationArray
    
    def quick_plot(self):
        bitmap = self.compute()
        colormap = "nipy_spectral" 
        ax = plt.axes()
        ax.set_aspect("equal")
        graph = ax.pcolormesh(self.xDomain, self.yDomain, bitmap, cmap=colormap)
        plt.colorbar(graph)
        plt.xlabel("Real-Axis")
        plt.ylabel("Imaginary-Axis")
        plt.show()



class NumbaMandelBrot(BaseLineMandelbrot):
    def __init__(self, xmin=-2, xmax=1, ymin=-1.5, ymax=1.5, num_points=2000, max_iterations=500, bound=2):
        super().__init__(xmin, xmax, ymin, ymax, num_points, max_iterations, bound)
        self.mandelbrot_calculate = njit(self.mandelbrot_calculate)
        self.compute_pixel = njit(self.compute_pixel)
        self.compute = njit(self.compute)


# @jitclass(spec=[('xmin', int32), ('xmax', int32), ('ymin', int32), ('ymax', int32), ('num_points', int32), ('max_iterations', int32), ('bound', int32), ('xDomain', np.linspace), ('yDomain', np.linspace, ('iterationArray', np.array))])
class NumbaMandelbrot:
    def __init__(self, xmin=-2, xmax=1, ymin=-1.5, ymax=1.5, num_points=2000, max_iterations=500, bound=2):
        self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
        self.num_points, self.max_iterations, self.bound = num_points, max_iterations, bound
        self.xDomain = np.linspace(xmin, xmax, num_points)
        self.yDomain = np.linspace(ymin, ymax, num_points)
        self.iterationArray = np.zeros((num_points, num_points), dtype=int)
    
    
    @staticmethod
    @njit
    def mandelbrot_calculate(z, p, c):
        return z**p + c

    
    @staticmethod
    @njit
    def compute_pixel(c, max_iterations, bound=2):
        z = 0
        p = 2
        for iteration in range(max_iterations):
            if abs(z) >= bound:
                return iteration
            z = NumbaMandelbrot.mandelbrot_calculate(z, p, c)
            
        return max_iterations
    

    @staticmethod
    @njit
    def compute(iterationArray, xDomain, yDomain, max_iterations):
        for y_i in range(len(yDomain)):
            for x_i in range(len(xDomain)):
                c = complex(xDomain[x_i], yDomain[y_i])
                iterationArray[y_i, x_i] = NumbaMandelbrot.compute_pixel(c, max_iterations)
        return iterationArray
    

    def quick_plot(self):
        bitmap = NumbaMandelbrot.compute(self.iterationArray, self.xDomain, self.yDomain, self.max_iterations)
        colormap = "nipy_spectral" 
        ax = plt.axes()
        ax.set_aspect("equal")
        graph = ax.pcolormesh(self.xDomain, self.yDomain, bitmap, cmap=colormap)
        plt.colorbar(graph)
        plt.xlabel("Real-Axis")
        plt.ylabel("Imaginary-Axis")
        plt.show()



def main():
    params = {'xmin': -2, 'xmax' : 1, 'ymin': -1.5, 'ymax' : 1.5, 'num_points' : 100, 'max_iterations' : 100, 'bound' : 2}
    # basic_mandel = BaseLineMandelbrot(**params)
    # basic_mandel.quick_plot()

    numba_mandel = NumbaMandelbrot(**params)
    numba_mandel.quick_plot()

    

main()
