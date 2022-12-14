# coco-optimization

Optimization results for test function (F1) with 5 constraints:

Blue point: Initial guess
Red point: solution 

<img title="a title" alt="Alt text" src="https://github.com/Erfan-Mashayekh/coco-optimization/blob/main/images/Figure_1.png">

Results for test function (F2) with 5 constraints:

<img title="a title" alt="Alt text" src="https://github.com/Erfan-Mashayekh/coco-optimization/blob/main/images/Figure_2.png">


TODO:
1. Add the rest of the test functions
2. Generalize the process of biases `b` generation in `compute_g` function
3. Read inputs from the input.json


Error Raiesd for using `Scipy.minimize()` for minimizing `n` points in paralell: 

- DeprecationWarning: Use of `minimize` with `x0.ndim != 1` is deprecated. Currently, singleton dimensions will be removed from `x0`, but an error will be raised in SciPy 1.11.0. opti_result = minimize(object_function,


