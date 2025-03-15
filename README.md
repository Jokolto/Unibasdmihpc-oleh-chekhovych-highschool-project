## Project Info ##

The goal of this project is to create a mandelbrot set visualisation with python, optimize it with use of parallel programming, and compare different methods of optimisation. 
To achieve that goal, the project utilises PyOMP and Numba. Later omp4py could also be tested.

## Set up information ##
Parallelized version of code only can be run on linux-64 (x86_64), osx-arm64 (mac), linux-arm64, and linux-ppc64le architectures, using conda due to PyOMP limitations. More about Pyomp: [github](https://github.com/Python-for-HPC/PyOMP), [documentation](https://pyomp.readthedocs.io/en/latest/). Paper about PyOMP can also be found in literature folder.


To run the code, following steps are required:

* Set up conda enviroment using python=3.9, [guide](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html)
* Install necessary packeges, through `conda install -c python-for-hpc -c conda-forge pyomp`.
* Clone git repository/code to your machine

## Authors ##
Author: Oleh Chekhovych, Supervisor: Reto Krummenacher
