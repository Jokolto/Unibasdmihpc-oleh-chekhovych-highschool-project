import numpy as np
import matplotlib.pyplot as plt
from omp4py import *
import time
import csv
import sysconfig


# setting parameters 
num_threads = 12
xmin, xmax, ymin, ymax = -2, 1, -1.5, 1.5
num_points = 1000  # amount of points between xmin and xmax (also ymin and ymax)
max_iterations = 500  # any positive integer value
xDomain, yDomain = np.linspace(xmin, xmax, num_points), np.linspace(ymin, ymax, num_points)


def mandelbrot_calculate(z, p, c):
    return z**p + c

def measure_performance(function, n_repeats, *args, **kwargs):
    t1 = time.perf_counter()
    for _ in range(n_repeats):
        function(*args, **kwargs)
        print(_)
    t2 = time.perf_counter()
    return (t2-t1)/n_repeats  


# computing 2-d array to represent the mandelbrot-set

@omp
def compute_points(xmin, xmax, ymin, ymax, num_points, max_iterations):
    xDomain, yDomain = np.linspace(xmin, xmax, num_points), np.linspace(ymin, ymax, num_points)
    bound = 2 
    iterationArray = np.zeros((num_points, num_points), dtype=int)

    with omp('parallel num_threads(12)'):
        with omp('for schedule(auto)'):
            for y_i in range(len(yDomain)):
                for x_i in range(len(xDomain)):
                    z = 0
                    p = 2
                    c = complex(xDomain[x_i], yDomain[y_i])

                    for iterationNumber in range(max_iterations):
                        if abs(z) >= bound:
                            iterationArray[y_i, x_i] = iterationNumber
                            break
                        else:
                            try:
                                z = mandelbrot_calculate(z, p, c)
                            except ValueError:
                                z = c
                            except ZeroDivisionError:
                                z = c
                    else:
                        iterationArray[y_i, x_i] = 0

    return iterationArray


# plotting the data
def plot_mandel():
    iterationArray = compute_points(xmin, xmax, ymin, ymax, num_points, max_iterations)
    colormap = "nipy_spectral" 
    ax = plt.axes()
    ax.set_aspect("equal")
    graph = ax.pcolormesh(xDomain, yDomain, iterationArray, cmap=colormap)
    plt.colorbar(graph)
    plt.xlabel("Real-Axis")
    plt.ylabel("Imaginary-Axis")
    # Save figure in PDF format for better scaling
    # plt.savefig("my_plot.pdf", dpi=300, bbox_inches="tight") 
    plt.show()

# plot_mandel()

def create_sheet_data():
    arguments = (xmin, xmax, ymin, ymax, num_points, max_iterations)
    avg_run_time = measure_performance(compute_points, 10, *arguments)
    print(f'average runtime in sec: {avg_run_time} with {num_threads} threads')
    with open("runtime_data.csv", 'a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['number of threads', 'average runtime in seconds'])
        writer.writerow({'number of threads': num_threads, 'average runtime in seconds': avg_run_time})

t1 = time.perf_counter()
compute_points(xmin, xmax, ymin, ymax, num_points, max_iterations)
t2 = time.perf_counter()

print(t2-t1)
print(sysconfig.get_config_var("Py_GIL_DISABLED"))

#create_sheet_data()