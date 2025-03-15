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

# this script contains a lot of repetetive code, because pyomp @njit does not work on methods and also cannot be turned off adequately once it decorates the function



def generate_parameters(xmin=-2, xmax=1, ymin=-1.5, ymax=1.5, num_points=2000, max_iterations=500, bound=2, use_numba=True, use_omp=True, num_threads=24, **kwargs):
    return {
        "xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax,
        "num_points": num_points, "max_iterations": max_iterations,
        "bound": bound, "use_numba": use_numba, "use_omp": use_omp and use_numba,
        "num_threads": num_threads if use_omp else 1,
        "xDomain": np.linspace(xmin, xmax, num_points),
        "yDomain": np.linspace(ymin, ymax, num_points),
        "iterationArray": np.zeros((num_points, num_points), dtype=int)
    }



@njit
def mandelbrot_pixel(c, max_iterations):
    z = 0
    for iterationNumber in range(max_iterations):
        if abs(z) >= 2:
            return iterationNumber
        z = z**2 + c

    return max_iterations



@njit
def compute_points(xDomain, yDomain, max_iterations, iterationArray, use_omp, num_threads, chunk_size='default', work=np.array([])):
    
    if use_omp:
        
        omp_set_num_threads(num_threads)
        
        if chunk_size == 'default':
            with omp('parallel'):
                with omp('for'):
                    for y_i in range(len(yDomain)):
                        for x_i in range(len(xDomain)):
                            c = complex(xDomain[x_i], yDomain[y_i])
                            z = mandelbrot_pixel(c, max_iterations)
                            work[omp_get_thread_num()] += z
                            iterationArray[y_i, x_i] = z
        else:    
            with omp('parallel'):
                with omp('for schedule(static, 1)'):
                    for y_i in range(len(yDomain)):
                        for x_i in range(len(xDomain)):
                            c = complex(xDomain[x_i], yDomain[y_i])
                            z = mandelbrot_pixel(c, max_iterations)
                            work[omp_get_thread_num()] += z
                            iterationArray[y_i, x_i] = z
    else:

        for y_i in range(len(yDomain)):
            for x_i in range(len(xDomain)):
                c = complex(xDomain[x_i], yDomain[y_i])
                numi = mandelbrot_pixel(c, max_iterations)
                iterationArray[y_i, x_i] = numi
                work[0] += numi

        
    return iterationArray, work



def p_mandelbrot_pixel(c, max_iterations):
    z = 0
    for iterationNumber in range(max_iterations):
        if abs(z) >= 2:
            return iterationNumber
        z = z**2 + c

    return max_iterations



def p_compute_points(xDomain, yDomain, max_iterations, iterationArray, *args):
    work = np.array([0])
    for y_i in range(len(yDomain)):
        for x_i in range(len(xDomain)):
            c = complex(xDomain[x_i], yDomain[y_i])
            numi = p_mandelbrot_pixel(c, max_iterations)
            iterationArray[y_i, x_i] = numi
            work[0] += numi
    return iterationArray, work



def main():
    config_test = {"xmin": -2, "xmax": 1, "ymin": -1.5, "ymax": 1.5, "num_points": 2000, "max_iterations": 500, "bound": 2, "use_numba": False, "use_omp": False, "num_threads": 1}
    params = generate_parameters(**config_test)

    bitmap = compute_points(params['xDomain'], params['yDomain'], params['max_iterations'], params['iterationArray'], params['use_omp'], params['num_threads'])


def measure_python(data_dict: dict, params: dict):

    for repeat in range(params['num_repeats']):
        params['use_numba'] = False
        params['use_omp'] = False
        run_params = generate_parameters(**params)
        params.update(run_params)

        t1 = time.perf_counter()
        bitmap, work = p_compute_points(run_params['xDomain'], run_params['yDomain'], run_params['max_iterations'], run_params['iterationArray'], run_params['use_omp'], run_params['num_threads'])
        runtime = time.perf_counter() - t1

        params['chunk_size'] = ('default')
        params['repeat'] = repeat + 1
        params['runtime'] = runtime
        params['comptime'] = 0
        params['work'] = work
        for k, v in params.items():
            if k in ("application", "num_points", 'max_iterations', 'num_threads', 'chunk_size', "repeat", "runtime", 'comptime', 'work'):
                data_dict[k].append(v)
    return bitmap

def measure_numba(data_dict, params):
    params['use_numba'] = True
    params['use_omp'] = False
    run_params = generate_parameters(**params)
    params.update(run_params)

    t1 = time.perf_counter()
    bitmap = compute_points(run_params['xDomain'], run_params['yDomain'], run_params['max_iterations'], run_params['iterationArray'], run_params['use_omp'], run_params['num_threads'])
    runandcomptime = time.perf_counter() - t1

    for repeat in range(params['num_repeats']):
        
        work = np.array([0])
        t1 = time.perf_counter()
        bitmap, work = compute_points(run_params['xDomain'], run_params['yDomain'], run_params['max_iterations'], run_params['iterationArray'], run_params['use_omp'], run_params['num_threads'], work=work)
        runtime = time.perf_counter() - t1

        params['chunk_size'] = ('default')
        params['repeat'] = repeat + 1
        params['runtime'] = runtime
        params['comptime'] = runandcomptime - runtime
        params['work'] = work
        for k, v in params.items():
            if k in ("application", "num_points", 'max_iterations', 'num_threads', 'chunk_size', "repeat", "runtime", 'comptime', 'work'):
                data_dict[k].append(v)
    
    return bitmap


def measure_omp(data_dict, params):

    params['use_numba'] = True
    params['use_omp'] = True


    for chunk_size in params['chunks_combinations']:
        params['chunk_size'] = chunk_size
        for thread in range(1, params['threads_range']+1):
            params['num_threads'] = thread
            run_params = generate_parameters(**params)
            params.update(run_params)

            work = np.zeros(thread, dtype=int)
            t1 = time.perf_counter()
            bitmap, work = compute_points(run_params['xDomain'], run_params['yDomain'], run_params['max_iterations'], run_params['iterationArray'], run_params['use_omp'], run_params['num_threads'], params['chunk_size'], work=work)
            compandruntime = time.perf_counter() - t1
            
            for repeat in range(params['num_repeats']):
                
                work = np.zeros(thread, dtype=int)
                t1 = time.perf_counter()
                bitmap, work = compute_points(run_params['xDomain'], run_params['yDomain'], run_params['max_iterations'], run_params['iterationArray'], run_params['use_omp'], run_params['num_threads'], params['chunk_size'], work=work)
                runtime = time.perf_counter() - t1

                params['repeat'] = repeat + 1
                params['runtime'] = runtime
                params['comptime'] = compandruntime - runtime
                params['work'] = work
                for k, v in params.items():
                    if k in ("application", "num_points", 'max_iterations', 'num_threads', 'chunk_size', "repeat", "runtime", 'comptime', 'work'):
                        data_dict[k].append(v)
            
    return bitmap



def test():
    applications = ('python', 'withnumba', 'withomp')
    measure_funcs = {'python': measure_python, 'withnumba': measure_numba, 'withomp': measure_omp}

    num_points_combinations = (100, 500, 2000, 5000)
    max_iteratins_combinations = (100, 500, 1000)

    threads_range = 8
    chunks_combinations = ('default', '1')
    num_repeats = 5

    constant_params = {"xmin": -2, "xmax": 1, "ymin": -1.5, "ymax": 1.5, "bound": 2}

    # data_dict = {'application': [], 'num_repeat': [], 'num_points': [], 'max_iterations': [], 'num_threads': [], 'chunk_size': [], 'runtime': [], 'xmin': [], 'xmax': [], 'ymin': [], 'ymax': [], 'bound': []}
    data_dict = defaultdict(list)


    for application in applications:
        for num_points in num_points_combinations:
            for max_iterations in max_iteratins_combinations:
                params = {"num_repeats": num_repeats, "num_points": num_points, "max_iterations": max_iterations, "threads_range": threads_range, "chunks_combinations": chunks_combinations, "application": application}
                params.update(constant_params)
                measure_funcs[application](data_dict, params)
                print(f"Experiment app={application}, res={num_points}, maxiter={max_iterations} done")
                # plot_mandel(bitmap, params)
                
    return data_dict


if __name__ == '__main__':
    dictdata = test()
    for k, v in dictdata.items():
        print(k, len(v))
    # print(dictdata)
    data = pd.DataFrame.from_dict(dictdata)
    data.to_csv('bit/data/bigdata.csv')
    print(data)
