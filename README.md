# Optimization of Constrained COCO Test Functions
This code is designed to test and compare optimization algorithms on the [COCO/BBOB functions](https://numbbo.github.io/coco/testsuites) with and without constraints. The objective function is provided by [IOH](https://iohprofiler.github.io/IOHexp/). The extension to constrained problems closely follows the [description of the bbob-constrained test suite](http://numbbo.github.io/coco-doc/bbob-constrained/functions.pdf). It is implemented in the COCOTestFunction class, which is imported from the pre_postprocessing.test_functions_coco module. As placeholder for eventual optimization algorithms, the SLSQP algorithm from the scipy.optimize package is used.

# Requirements
This code requires the following Python packages to be installed:
- IOHExperimenter 0.3.8 (https://iohprofiler.github.io/IOHexperimenter/)
- numpy
- matplotlib
- scipy

# How to Use
The code reads the settings for the optimization problem from a JSON file called model.json. The file must be in the same directory as the main code. The JSON file must contain the following fields:

- function_type: Type of test function. Currently only COCO test functions are supported, with value 1.
- coco_id: ID of the COCO test function to use.
- instance: Instance ID of the IOH problem.
- dimensions: Input dimension of the test function. Must be one of {2,3,5,10,20,40}.
- constrained: Boolean indicating whether the optimization problem has constraints or not.
- num_constraints: Number of constraints, if constrained is True.
- lower_bounds: Lower bounds of the input variables.
- upper_bounds: Upper bounds of the input variables.
- seed: Seed for the random number generator used to generate the initial and optimum points.
- opt_read: Boolean indicating whether to read initial points from a CSV file called data.csv.
- grid_size: Grid size for plotting the test function.

To use the code, simply run the main script. The optimized solution and optimization error will be printed to the console, and a contour plot of the test function will be displayed if the input dimension is 2.

# Vectorization
The code also includes a test of the vectorization ability of the objective and constraint functions. A random set of input vectors is generated, and the functions are applied to these vectors using vectorized operations. If the functions cannot handle vectorized inputs, a warning is issued.
