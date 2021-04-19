from copy import deepcopy
import utilsforminds
import functools
import numpy as np
import utilsforminds.helpers as helpers
from inspect import signature
from utilsforminds.containers import Grid

def signature_multi_binding(new_old_signature_dict):
    def decorator(func):
        @functools.wraps(func)
        def decorator_inner(*args, **kwargs):
            processed_news = []
            for new, old in new_old_signature_dict.items():
                raise Exception(NotImplementedError)

def redirect_function(module, func_name = None):
    """Redirect the function to another function in another module"""

    def decorator_redirect_function(func):
        @functools.wraps(func)
        def wrapper_redirect_function(*args, **kwargs):
            if func_name is None:
                return getattr(module, func.__name__)(*args, **kwargs)
            else:
                return getattr(module, func_name)(*args, **kwargs)
        return wrapper_redirect_function
    return decorator_redirect_function

def check_bad_values_in_numpy_arr(check_list = ['nan', 'inf']):
    """Check whether input Numpy arguments contain invalid values
    
    Examples
    --------
    test_arr_1 = np.ones((3, 3))
    test_arr_2 = np.ones((3, 3)) + 2.
    test_arr_2[1, 2] = np.inf
    print(add_arr(test_arr_1, name = 'hello', arr_2= test_arr_2))
        :Exception has occurred: Exception
        Invalid values found
    """

    def decorator_check_bad_values_in_numpy_arr(func):
        @functools.wraps(func)
        def wrapper_check_bad_values_in_numpy_arr(*args, **kwargs):
            for arg in args:
                if isinstance(arg, np.ndarray):
                    bool_arr = np.full(arg.shape, False, dtype= bool)
                    if 'nan' in check_list:
                        bool_arr = bool_arr + np.isnan(arg)
                    if 'inf' in check_list:
                        bool_arr = bool_arr + np.isinf(arg)
                    if np.any(bool_arr):
                        indices_arr = np.argwhere(bool_arr)
                        for i in range(min(10, indices_arr.shape[0])):
                            print(f"WARNING: Invalid value: {helpers.index_arr_with_arraylike(arg, indices_arr[i])} is found at indices: {indices_arr[i]}")
                        raise Exception("Invalid values found")
            for arg in kwargs.values():
                if isinstance(arg, np.ndarray):
                    bool_arr = np.full(arg.shape, False, dtype= bool)
                    if 'nan' in check_list:
                        bool_arr = bool_arr + np.isnan(arg)
                    if 'inf' in check_list:
                        bool_arr = bool_arr + np.isinf(arg)
                    if np.any(bool_arr):
                        indices_arr = np.argwhere(bool_arr)
                        for i in range(min(10, indices_arr.shape[0])):
                            print(f"WARNING: Invalid value: {helpers.index_arr_with_arraylike(arg, indices_arr[i])} is found at indices: {indices_arr[i]}")
                        raise Exception("Invalid values found")
            return func(*args, **kwargs)
        return wrapper_check_bad_values_in_numpy_arr
    return decorator_check_bad_values_in_numpy_arr

def grid_of_functions(param_to_grid, param_formatter_dict = None, grid_condition = None):
    """Deep copy kwargs.
    
    Examples 1
    ----------
    test_grid = Grid(1, 2, 4)

    @grid_of_functions(param_to_grid= "number", param_formatter_dict= {"path": lambda **kwargs: kwargs["path"] + str(kwargs["number"])})
    def test_func(path = "here.txt", number = 4, dummy = 77):
        print(path)
        print(number + 5)
        print(dummy)
        return number

    result = test_func(path = "here.txt", number = test_grid, dummy = 77)
    print(result)

    >>> here.txt1
    6
    77
    here.txt2
    7
    77
    here.txt4
    9
    77
    [1, 2, 4]

    Examples 2
    ----------
    test_grid = Grid(1, 2, 4)

    @grid_of_functions(param_to_grid= "number", param_formatter_dict= {"path": lambda **kwargs: kwargs["path"] + str(kwargs["number"])})
    def test_func(path = "here.txt", number = 4, dummy = 77):
        print(path)
        print(number + 5)
        print(dummy)
        return number

    result = test_func()
    print(result)
    >>> here.txt
    9
    77
    4
    """

    # if not isinstance(list_of_params_to_grid, list):
    #     list_of_params_to_grid = [list_of_params_to_grid]
    if param_formatter_dict is None:
        param_formatter_dict = {}
    if grid_condition is None:
        grid_condition = lambda **kwargs: True
    def decorator_grids(func):
        @functools.wraps(func)
        def wrapper_grids(*args, **kwargs):
            if param_to_grid in kwargs.keys() and len(param_formatter_dict) > 0 and any([param in kwargs.keys() for param in param_formatter_dict.keys()]) and (isinstance(kwargs[param_to_grid], Grid)) and grid_condition(**kwargs):
                kwargs_copy = deepcopy(kwargs)
                returns_list = []
                for component in kwargs[param_to_grid].list_of_components:
                    kwargs_copy[param_to_grid] = component
                    for param in param_formatter_dict.keys(): 
                        kwargs_copy[param] = deepcopy(kwargs[param])
                    for param in param_formatter_dict.keys():
                        kwargs_copy[param] = param_formatter_dict[param](**kwargs_copy)
                    returns_list.append(func(*args, **kwargs_copy))
                return returns_list
            else:
                return func(*args, **kwargs)
        return wrapper_grids
    return decorator_grids




if __name__ == "__main__":
    pass
    # test_dict = {"a": 3, "b": 5}
    # print(test_dict.update({"a": 9}))
    # print(test_dict)

    test_grid = Grid(1, 2, 4)

    @grid_of_functions(param_to_grid= "number", param_formatter_dict= {"path": lambda **kwargs: kwargs["path"] + str(kwargs["number"])})
    def test_func(path = "here.txt", number = 4, dummy = 77):
        print(path)
        print(number + 5)
        print(dummy)
        return number

    result = test_func()
    print(result)