import utilsforminds
import functools

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
