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

def get_value_of_proportional_rank(amount_arr, mask_arr, proportional_rank):
    """Get the value corresponding to the given rank.
    
    The rank is counted from the smallest value, so the proportional_rank 0.7 of 1~10 is 7.

    Examples
    --------
    amount_arr = np.array([[1., 2., 3.], [4., 5., 6.]])\n
    mask_arr = np.array([[1., 1., 0.], [1., 1., 1.]])\n
    print(get_value_of_proportional_rank(amount_arr = amount_arr, mask_arr = mask_arr, proportional_rank = 0.7))
        : 5.0
    print(get_value_of_proportional_rank(amount_arr = amount_arr, mask_arr = mask_arr, proportional_rank = 0.3))
        : 2.0
    """

    assert(0. <= proportional_rank and proportional_rank <= 1.)
    mask_arr_clean = np.where(mask_arr >= 1., 1., 0.)
    numEntriesImputed = np.count_nonzero(mask_arr_clean)
    if numEntriesImputed > 0:
        rank = int(numEntriesImputed * proportional_rank) ## values higher than the rank from the 'smallest' value will survive, in other words maskKeepThreshold * 100% will die.
        rankValue = np.partition(amount_arr[np.nonzero(mask_arr_clean)], rank)[rank]
        return rankValue
    else:
        return 1e-16

if __name__ == "__main__":
    amount_arr = np.array([[1., 2., 3.], [4., 5., 6.]])
    mask_arr = np.array([[1., 1., 0.], [1., 1., 1.]])
    print(get_value_of_proportional_rank(amount_arr = amount_arr, mask_arr = mask_arr, proportional_rank = 0.3))