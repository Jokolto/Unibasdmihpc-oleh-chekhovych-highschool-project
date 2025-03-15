import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
import csv
from numba import njit, prange
from numba.openmp import openmp_context as omp
from numba.openmp import omp_get_num_threads, omp_set_num_threads, omp_get_thread_num
from PIL import Image
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# setting parameters 

xmin, xmax, ymin, ymax = -2, 1, -1.5, 1.5
num_points = 5000  # amount of points between xmin and xmax (also ymin and ymax)
max_iterations = 1000 


bound = 2 
use_numba = True
use_omp = True
num_threads = 24

display_set = True
measure_work = False
schedule = 'static1'

# csv data creation
csv_threads_amount = 24


# some calculations
xDomain, yDomain = np.linspace(xmin, xmax, num_points), np.linspace(ymin, ymax, num_points)
iterationArray = np.zeros((num_points, num_points), dtype=int)
if not use_omp:
    num_threads = 1
arguments = (xDomain, yDomain, max_iterations, iterationArray, num_threads)

# cant have omp without numba, checking just in case
use_omp = use_numba and use_omp

# disable njit if numba is not enabled 
njit = njit if use_numba else lambda f: f



@njit
def compute_points(xDomain, yDomain, max_iterations, iterationArray, num_threads, use_omp=use_omp):    # iterationArray is a parameter of this function due to issue of numba compiler. (similiar issue https://stackoverflow.com/questions/71902946/numba-no-implementation-of-function-functionbuilt-in-function-getitem-found)
    # iterationArray = np.zeros((num_points, num_points), dtype=int)
    if use_omp:
        work = {i:0 for i in range(num_threads)}
        omp_set_num_threads(num_threads)
        with omp('parallel'):
            with omp('for schedule(static, 2)'):
                for y_i in range(len(yDomain)):
                    # print(y_i, omp_get_thread_num())
                    for x_i in range(len(xDomain)):
                        c = complex(xDomain[x_i], yDomain[y_i])
                        z = mandelbrot_pixel(c, max_iterations)
                        if measure_work:
                            work[omp_get_thread_num()] += z
                        
                        iterationArray[y_i, x_i] = z

    else:      # use pure python if omp is disabled
        work = {0:0}
        for y_i in range(len(yDomain)):
            for x_i in range(len(xDomain)):
                c = complex(xDomain[x_i], yDomain[y_i])
                z = mandelbrot_pixel(c, max_iterations)
                iterationArray[y_i, x_i] = z

        

    return iterationArray, work

@njit
def mandelbrot_pixel(c, max_iterations):
    z = 0
    for iterationNumber in range(max_iterations):
        if abs(z) >= bound:
            return iterationNumber
        z = z**2 + c

    return max_iterations


def measure_performance(function, n_repeats, *args, **kwargs):
    # it takes time to compile the function to machine code, so we ignore first time execution
    ct1 = time.perf_counter()
    function(*args, **kwargs)
    ct2 = time.perf_counter()
    

    t1 = time.perf_counter()
    for repeat in range(n_repeats):
        res = function(*args, **kwargs)
        # print(f"function runned {repeat+1}-th time ")  feedback print, but it takes a bit of time to run as well
    t2 = time.perf_counter()

    run_time = (t2-t1)/n_repeats  
    compiled_time = (ct2 - ct1) - run_time
    return float(run_time)


def check_numba(use_numba=True):
    global compute_points, mandelbrot_calculate, mandelbrot_pixel
    if use_numba:
        compute_points = njit(compute_points)
        mandelbrot_pixel = njit(mandelbrot_pixel.py_func)
        
    else:
        compute_points = compute_points.py_func
        mandelbrot_pixel = mandelbrot_pixel.py_func


def apply_blue_palette(image_array, max_it):
    """
    Applies a predefined blue palette colormap to a 2D NumPy array.

    Parameters:
        image_array (np.ndarray): 2D array where each value represents the iteration count.
        max_it (int): Maximum iteration count, used to determine black pixels.

    Returns:
        np.ndarray: 3D RGB array (H, W, 3) with colored pixels.
    """
    mapping = {
        0: (66, 30, 15),   # brown 3
        1: (25, 7, 26),    # dark violet
        2: (9, 1, 47),     # darkest blue
        3: (4, 4, 73),     # blue 5
        4: (0, 7, 100),    # blue 4
        5: (12, 44, 138),  # blue 3
        6: (24, 82, 177),  # blue 2
        7: (57, 125, 209), # blue 1
        8: (134, 181, 229),# blue 0
        9: (211, 236, 248),# lightest blue
        10: (241, 233, 191), # lightest yellow
        11: (248, 201, 95),  # light yellow
        12: (255, 170, 0),   # dirty yellow
        13: (204, 128, 0),   # brown 0
        14: (153, 87, 0),    # brown 1
        15: (106, 52, 3)     # brown 2
    }

    height, width = image_array.shape
    rgb_array = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            i = image_array[y, x]
            rgb_array[y, x] = (0, 0, 0) if i == max_it else mapping[i % 16]

    return rgb_array



def apply_gray_palette(image_array, max_it):
    """
    Applies a very subtle grayscale colormap to a 2D NumPy array, designed for background purposes.

    Parameters:
        image_array (np.ndarray): 2D array where each value represents the iteration count.
        max_it (int): Maximum iteration count, used to determine background pixels.

    Returns:
        np.ndarray: 3D RGB array (H, W, 3) with lightly shaded pixels.
    """
    # Define a very subtle grayscale palette (just shades of gray from white to light gray)
    mapping = {
        0: (255, 255, 255),    # white
        1: (245, 245, 245),    # very light gray
        2: (235, 235, 235),    # light gray 1
        3: (225, 225, 225),    # light gray 2
        4: (215, 215, 215),    # light gray 3
        5: (205, 205, 205),    # light gray 4
        6: (195, 195, 195),    # light gray 5
        7: (185, 185, 185),    # light gray 6
        8: (175, 175, 175),    # light gray 7
        9: (165, 165, 165),    # light gray 8
        10: (155, 155, 155),   # light gray 9
        11: (145, 145, 145),   # light gray 10
        12: (135, 135, 135),   # light gray 11
        13: (125, 125, 125),   # light gray 12
        14: (115, 115, 115),   # light gray 13
        15: (105, 105, 105)    # light gray 14
    }

    height, width = image_array.shape
    rgb_array = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            i = image_array[y, x]
            rgb_array[y, x] = mapping[i % 16] if i < max_it else (255, 255, 255)  # white for max iteration pixels

    return rgb_array



def apply_border_palette(image_array, max_it, threshold=30):
    """
    Applies a grayscale colormap to highlight only the boundary of the Mandelbrot set.

    Parameters:
        image_array (np.ndarray): 2D array where each value represents the iteration count.
        max_it (int): Maximum iteration count, used to determine which points are part of the set.
        threshold (int): Iteration threshold for determining points near the boundary.

    Returns:
        np.ndarray: 3D RGB array (H, W, 3) with colors applied to the boundary.
    """
    # Define a subtle grayscale palette for boundary highlighting
    mapping = {
        0: (0, 0, 0),          # Inside the Mandelbrot set, black (or white for background)
        1: (255, 255, 255),    # Points inside the set, white (background)
    }

    # For points near the boundary (iterations close to max_it), assign subtle gray shades
    for i in range(threshold, max_it):
        gray_value = int(255 - (255 * (i - threshold) / (max_it - threshold)))
        mapping[i] = (gray_value, gray_value, gray_value)  # Subtle grayscale transition

    height, width = image_array.shape
    rgb_array = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            i = image_array[y, x]

            # For points near the boundary (close to max_it), apply the boundary palette
            if i >= (max_it - threshold):
                rgb_array[y, x] = mapping.get(i, (255, 255, 255))  # Default to white if i exceeds max_it
            else:
                rgb_array[y, x] = (255, 255, 255)  # Inside the set is pure white

    return rgb_array


def light_grayscale_colormap():
    # Define a custom colormap that goes from white to very light gray
    colors = [
        (1.0, 1.0, 1.0),  # white (background)
        (0.95, 0.95, 0.95),  # very light gray
        (0.9, 0.9, 0.9)  # slightly darker gray (but still light)
    ]
    return mcolors.LinearSegmentedColormap.from_list("light_gray", colors, N=256)


# plotting the data
def plot_mandel(bitmap, xDomain, yDomain, save=True):
    # colormap = "nipy_spectral"
    colormap = light_grayscale_colormap()
    plt.figure(figsize=(10, 8))
    
    # Create a heatmap using Seaborn, remove axis and ticks
    ax = sns.heatmap(bitmap, cmap=colormap, xticklabels=False, yticklabels=False, cbar_kws={'label': 'Iterations'})
    
    # Remove axis labels and ticks
    ax.set_axis_off()
    
    if save:
        # cmap = 'inferno'
        cmap = light_grayscale_colormap()
        rgb_array = apply_border_palette(bitmap, max_iterations)
        img = Image.fromarray(rgb_array)
        img.save(f'bit/plots/mandelbrot_generated_nump={num_points}_maxi={max_iterations}_border.pdf')
    
    # Show only the image, no extra plot controls
    plt.show()


def create_csv_data_runtime(): 
    fieldnames = ['n_threads', 'runtime', 'resolution', 'max_iterations']
    for threads in range(1, csv_threads_amount+1):
        arguments = (xDomain, yDomain, max_iterations, iterationArray, threads)
        avg_run_time = measure_performance(compute_points, 10, *arguments)

        print(f'({threads}) Writing into csv:')
        print(f'average runtime in sec: {avg_run_time} with {threads} threads, parameters=(num_points={num_points}, max_iterations={max_iterations})')
        print()

        with open(f"bit/data/runtime_data_numpoints={num_points}_maxiter={max_iterations}_schedule={schedule}.csv", 'a', newline='') as csv_file:
            
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writerow({'n_threads': threads, 'runtime': avg_run_time, 'resolution': (num_points, num_points), 'max_iterations': max_iterations})


def create_csv_data_comparison():
    fieldnames = ['titles', 'runtime']
    arguments = (xDomain, yDomain, max_iterations, iterationArray, num_threads)

    runtime_omp_static_schedule = measure_performance(compute_points, 10, *arguments)
    print('Measured runtime of function with omp')
    runtime_numba = measure_performance(compute_points, 10, use_omp=False, *arguments)
    print('Measured runtime of function with numba enabled, no omp')

    check_numba(use_numba=False)
    run_time_python = measure_performance(compute_points, 10, use_omp=False, *arguments)
    print('Measured runtime of function with python, no numba nor omp')
    
    
    comparison = {'Runtime with pure python': run_time_python,
                'Using numba compilator': runtime_numba,  
                'parallelizing with PyOmp with 24 threads': runtime_omp_static_schedule
    }

    
    for title, runtime in comparison.items():
        with open(f"data/runtime_comparison_params=(res={num_points}x{num_points}_maxi={max_iterations}_schedule={schedule}).csv", 'a', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writerow({'titles': title, 'runtime': runtime})



def plot_csv_threads_vs_runtime(data_path_csv: str, save=False):
    with open(data_path_csv, 'r', newline='') as csv_file:
        fieldnames = ['n_threads', 'runtime', 'resolution', 'max_iterations']
        csv_reader = csv.DictReader(csv_file, fieldnames = fieldnames)
        n_threads_axis, runtime_axis = [], []
        for row in csv_reader:
            n_threads_axis.append(row['n_threads'])
            runtime_axis.append(float(row['runtime']))
        

    title = f'''Number of threads and runtime correlation; 
            parameters=(num_points={num_points}, max_iterations={max_iterations})'''
    fig, ax = plt.subplots()
    ax.plot(n_threads_axis, runtime_axis)
    ax.set(xlabel='number of threads', ylabel='average runtime (s)', title=title)
    ax.grid()
    if save:
        plt.savefig(f'bit/plots/threads_runtime_plot_numpoints={num_points}_maxiter={max_iterations}_schedule={schedule}.png')
    plt.show()
    

def plot_dict(d:dict):
    lists = sorted(d.items())
    x, y = zip(*lists) # unpack a list of pairs into two tuples

    fig, ax = plt.subplots()
    ax.bar(x, y, width=1, edgecolor="white", linewidth=0.7)

    # Ensure all keys are displayed
    ax.set_xticks(x)  # Explicitly set x-ticks
    ax.set_xticklabels(x, rotation=45)  # Rotate labels if necessary

    # Labels and title
    ax.set_xlabel("Thread")
    ax.set_ylabel("Amount of iterations")
    ax.set_title("Iteration distribution among threads")

    plt.savefig(f'bit/plots/work_distribution/iteration_distribution_numthreads={num_threads}_schedule={schedule}')
    plt.show()
    

def main():
    t1 = time.perf_counter()
    bitmap, work = compute_points(xDomain=xDomain, yDomain=yDomain, max_iterations=max_iterations, iterationArray=iterationArray, num_threads=num_threads)
    t2 = time.perf_counter()

    print(f'Runtime in sec: {float(t2-t1)} with {num_threads} threads, parameters=(num_points={num_points}, max_iterations={max_iterations}, numba={use_numba}, omp={use_omp})')


    if display_set:
        # plot_dict(work)
        plot_mandel(bitmap, xDomain, yDomain)
    


if __name__ == '__main__':
    # create_csv_data_runtime()
    # create_csv_data_comparison()
    # plot_csv_threads_vs_runtime("bit/data/runtime_data_numpoints=2000_maxiter=500_schedule=default.csv", save=True)
    # print(measure_performance(compute_points, 10, *arguments))
    main()