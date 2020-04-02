import numpy as np
import utilsforminds.helpers as helpers
from copy import deepcopy
import numbers

def p_exponent_matrix(arr, p):
    """
    
    Examples
    --------
    self.diagonals['D_7'] = (self.p / 2) * helpers.p_exponent_matrix(W_p1p @ W_p1p.transpose() + self.delta * np.eye(self.d), (self.p / 2 - 1))
    """
    assert(len(arr.shape) == 2)

    #%% Using SVD
    u, s, v = np.linalg.svd(arr, full_matrices=False)
    return u @ np.diag(s ** p) @ v

    #$$ Using Igenvalue decomposition
    # w, v = np.linalg.eig(arr)
    # return v @ np.diag(w ** p) @ v.transpose()

def get_norm_from_matrix(arr, under_p_1, under_p_2):
    """
    
    Examples
    --------
    testArr_1 = np.array(list(range(6))).reshape(2,3)\n
    print(helpers.get_norm_from_matrix(testArr_1, 2, 2))
        => 7.416198487095663
    """

    summed = 0
    for i in range(arr.shape[0]):
        summed += np.sum(arr[i, :] ** under_p_1) ** (under_p_2 / under_p_1)
    return summed ** (1 / under_p_2)

def get_RMSE(arr_1, arr_2):
    assert(arr_1.shape == arr_2.shape)
    return (np.sum((arr_1 - arr_2) ** 2) / arr_1.shape[0]) ** (1/2.)

def statistics_across_containers(containers_list, kind_of_stat = 'std'):
    """Get statistic values across the list of the leaves of the nested containers whose shapes are same
    
    Parameters
    ----------
    containers_list : list
        List of containers, say [container_1, container_2, ..., container_n]. Here container_1, container_2, ..., container_n can be nested dict or list or tuple, but their nested structure should be same.
    
    Returns
    -------
    : container
        Container of statistic values

    Examples
    --------
    test_containers_list = [{'a': [1, 2.5, 3], 'b': {'c': [2, 4], 'd': 5}}, {'a': [2, 3.5, 4], 'b': {'c': [3, 5], 'd': 6}}]\n
    print(statistics_across_containers(test_containers_list, kind_of_stat = 'mean'))
        {'a': [1.5, 3.0, 3.5], 'b': {'c': [2.5, 4.5], 'd': 5.5}}
    """

    assert(isinstance(containers_list, list))
    result_container = deepcopy(containers_list[0])
    paths_to_leaves = deepcopy(helpers.get_paths_to_leaves(result_container))
    for path in paths_to_leaves:
        container_last_one = helpers.access_with_list_of_keys_or_indices(result_container, path[:-1])
        container_last_one[path[-1]] = []
    
    for container in containers_list:
        for path in paths_to_leaves:
            helpers.access_with_list_of_keys_or_indices(result_container, path).append(helpers.access_with_list_of_keys_or_indices(container, path))
    
    for path in paths_to_leaves:
        container_last_one = helpers.access_with_list_of_keys_or_indices(result_container, path[:-1])
        elements_across_containers_list = helpers.access_with_list_of_keys_or_indices(result_container, path)
        is_all_elements_are_number = True
        for element in elements_across_containers_list:
            assert(not isinstance(element, list))
            if not isinstance(element, numbers.Number):
                is_all_elements_are_number = False
                break
        if is_all_elements_are_number:
            if kind_of_stat == 'std':
                container_last_one[path[-1]] = np.std(np.array(elements_across_containers_list))
            elif kind_of_stat == 'mean':
                container_last_one[path[-1]] = np.mean(np.array(elements_across_containers_list))
            else:
                raise Exception(f"Unsupported kind_of_stat: {kind_of_stat}")
    
    return result_container

if __name__ == '__main__':
    pass
    # test_containers_list = [{'a': [1, 2.5, 3], 'b': {'c': [2, 4], 'd': 5}}, {'a': [2, 3.5, 4], 'b': {'c': [3, 5], 'd': 6}}]
    # print(statistics_across_containers(test_containers_list, kind_of_stat = 'mean'))