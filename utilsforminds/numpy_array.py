import numpy as np

def push_arr_to_range(nparr, vmin = None, vmax = None):
    if vmin is not None and vmax is not None:
        return np.where(nparr > vmax, vmax, np.where(nparr < vmin, vmin, nparr))
    elif vmin is not None and vmax is None:
        return np.where(nparr < vmin, vmin, nparr)
    elif vmin is None and vmax is not None:
        return np.where(nparr > vmax, vmax, nparr)
    else:
        # print("WARNING: vmin and vmax both are None, so nparr is not pushed to range.")
        return np.copy(nparr)
    