from copy import deepcopy

def get_items_from_list_conditionally(list_, condition_function, whether_deepcopy = False):
    """ This is reference-copy function not value-copy.
    
    Examples
    --------
    test_list = [{'a':1, 'b':2}, {'a':3, 'b':2}, {'a':2, 'b':4}]\n
    print(get_items_from_list_conditionally(test_list, lambda x: True if x['a'] >= 2 else False))
        [{'a': 3, 'b': 2}, {'a': 2, 'b': 4}]
    """

    assert(type(list_) == type([]))
    result_collection = []
    for item in list_:
        if condition_function(item):
            result_collection.append(deepcopy(item)) if whether_deepcopy else result_collection.append(item)
    return result_collection

def merge_two_dicts_leaves_rec(dict_1, dict_2):
    """ Merge two dictionaries recursively.
    Examples
    --------
    test_dict_1 = {"n": ["a", "b", "c"], "m": {"hi": 4}, 1: ["h", "i"], "dup": {"ace": {1: 2, 3: 4}}, "none": None}
    test_dict_2 = {"n": ["a", "d"], "m": {"hi" : 2, "hello": {3: 2}}, 2: 4, "dup": {"ace": {1: {4: 5, 6: 7}, 3: None}}, "none": 4}
    print(merge_two_dicts_leaves_rec(test_dict_1, test_dict_2))
        {'n': ['a', 'd'], 'm': {'hi': 2, 'hello': {3: 2}}, 2: 4, 'dup': {'ace': {1: {4: 5, 6: 7}, 3: None}}, 'none': 4, 1: ['h', 'i']}
    """

    if dict_1 is None:
        return dict_2
    if dict_2 is None:
        return dict_1
    assert(isinstance(dict_1, dict) and isinstance(dict_2, dict))
    merged_dict = {}
    for dict_2_key in dict_2.keys():
        if isinstance(dict_2[dict_2_key], dict) and dict_2_key in dict_1.keys() and isinstance(dict_1[dict_2_key], dict):
            merged_dict[dict_2_key] = merge_two_dicts_leaves_rec(dict_1[dict_2_key], dict_2[dict_2_key])
        else:
            merged_dict[dict_2_key] = dict_2[dict_2_key]
    for dict_1_key in dict_1.keys():
        if dict_1_key not in merged_dict.keys():
            merged_dict[dict_1_key] = dict_1[dict_1_key]
    return merged_dict
            

def merge_dictionaries(list_of_dicts : list, use_last_when_overlapped = True, recursive_overwritting = True):
    """ Merge dictionaries with 'update' method.
        Always deepcopy because of nature of dict.update method. -> Changed to shallowcopy 200910.

    Examples
    --------
    test_list = [{'a':1, 'b':2, 'c':9}, {'a':3, 'b':2}, {'a':2, 'b':4}]\n
    merge_dictionaries(test_list, use_last_when_overlapped = True)\n
        == {'a': 2, 'b': 4, 'c': 9}
    merge_dictionaries(test_list, use_last_when_overlapped = False)\n
        == {'a': 1, 'b': 2, 'c': 9}
    """
    
    merged_dict = {}
    if use_last_when_overlapped:
        range_to_iter = range(len(list_of_dicts))
    else:
        range_to_iter = range(len(list_of_dicts) -1, -1, -1)
    for dict_idx in range_to_iter:
        # if list_of_dicts[dict_idx] is not None:
        #     assert(type(list_of_dicts[dict_idx]) == type({}))
        #     merged_dict.update(list_of_dicts[dict_idx])
        merged_dict = merge_two_dicts_leaves_rec(merged_dict, list_of_dicts[dict_idx])
    return merged_dict

def merge_lists(list_of_lists : list, use_last_when_overlapped = True):
    """Merge lists through deepcopy

    Examples
    --------
    test_list = [["a", "b", "c"], ["d", "e", "f", "g"], ["h", "i"]]
    print(merge_lists(test_list, use_last_when_overlapped = True))
        == ['h', 'i', 'f', 'g']
    print(merge_lists(test_list, use_last_when_overlapped = False))
        == ['a', 'b', 'c', 'g']
    """

    merged_list = []
    if use_last_when_overlapped:
        range_to_iter = range(len(list_of_lists) -1, -1, -1)
    else:
        range_to_iter = range(len(list_of_lists))
    for list_idx in range_to_iter:
        if list_of_lists[list_idx] is not None:
            assert(type(list_of_lists[list_idx]) == type([]))
            for list_idx_idx in range(len(merged_list), len(list_of_lists[list_idx])): ## append only when there are more elements in list_of_lists[list_idx].
                merged_list.append(list_of_lists[list_idx][list_idx_idx])
    return merged_list

def copy_dict_and_delete_element(dict_to_copy, list_of_keys_to_delete):
    """
    
    Examples
    --------
    test_dict = {"n": ["a", "b", "c"], "m": ["d", "e", "f", "g"], 1: ["h", "i"]}
    print(copy_dict_and_delete_element(dict_to_copy = test_dict, list_of_keys_to_delete = ["n", 1]))
        == {'m': ['d', 'e', 'f', 'g']}
    """

    result = deepcopy(dict_to_copy)
    for key in list_of_keys_to_delete:
        if key in result.keys():
            del result[key]
    return result

def squeeze_list_of_numbers_with_average_of_each_range(list_of_numbers, num_points_in_list_out = 100):
    """
    
    Examples
    --------
    test_list = list(range(5))
    print(squeeze_list_of_numbers_with_average_of_each_range(test_list, num_points_in_list_out= 2))
        >>> [0.5, 2.5]
    test_list = list(range(20))
    print(squeeze_list_of_numbers_with_average_of_each_range(test_list, num_points_in_list_out= 5))
        >>> [1.5, 5.5, 9.5, 13.5, 17.5]
    test_list = list(range(19))
    print(squeeze_list_of_numbers_with_average_of_each_range(test_list, num_points_in_list_out= 5))
        >>> [1.0, 4.0, 7.0, 10.0, 13.0]
    """

    if len(list_of_numbers) <= num_points_in_list_out:
        return deepcopy(list_of_numbers)
    else:
        list_out = []
        num_numbers_in_each_group = len(list_of_numbers) // num_points_in_list_out
        for group_idx in range(num_points_in_list_out):
            average = sum(list_of_numbers[group_idx * num_numbers_in_each_group: (group_idx + 1) * num_numbers_in_each_group]) / num_numbers_in_each_group
            list_out.append(average)
        return list_out

if __name__ == "__main__":
    pass
    # test_list = list(range(5))
    # print(squeeze_list_of_numbers_with_average_of_each_range(test_list, num_points_in_list_out= 2))

    # test_list = list(range(20))
    # print(squeeze_list_of_numbers_with_average_of_each_range(test_list, num_points_in_list_out= 5))

    # test_list = list(range(19))
    # print(squeeze_list_of_numbers_with_average_of_each_range(test_list, num_points_in_list_out= 5))