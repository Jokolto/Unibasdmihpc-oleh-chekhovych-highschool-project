import numpy as np
import matplotlib.pyplot as plt
import time
import csv
from numba import njit, prange
from numba.openmp import openmp_context as omp
from numba.openmp import omp_get_num_threads, omp_set_num_threads, omp_get_thread_num, omp_get_wtime
from collections import defaultdict
import pandas as pd
from numba.typed import Dict
from numba.core import types




def mandelbrot_pixel(c, max_iterations):
    z = 0
    for iterationNumber in range(max_iterations):
        if abs(z) >= 2:
            return iterationNumber
        z = z**2 + c
    return max_iterations


def compute_points(xDomain, yDomain, max_iterations, iterationArray):
    for y in yDomain:
        for x in xDomain:
            c = complex(x, y)
            iterationArray[y, x] = mandelbrot_pixel(c, max_iterations)
    return iterationArray 

@njit
def mandelbrot_pixel(c, max_iterations):
    z = 0
    for iterationNumber in range(max_iterations):
        if abs(z) >= 2:
            return iterationNumber
        z = z**2 + c
    return max_iterations 

@njit
def compute_points(xDomain, yDomain, max_iterations, iterationArray):
    for y_i in range(len(yDomain)):
        for x_i in range(len(xDomain)):
            c = complex(xDomain[x_i], yDomain[y_i])
            iterationArray[y_i, x_i] = mandelbrot_pixel(c, max_iterations) 
    return iterationArray 




@njit
def compute_points(xDomain, yDomain, max_iterations, iterationArray, num_threads):
    omp_set_num_threads(num_threads)
    with omp('parallel'):
        with omp('for schedule(static, 1)'):
            for y_i in range(len(yDomain)):
                for x_i in range(len(xDomain)):
                    c = complex(xDomain[x_i], yDomain[y_i])
                    z = mandelbrot_pixel(c, max_iterations)
                    iterationArray[y_i, x_i] = z
    return iterationArray
