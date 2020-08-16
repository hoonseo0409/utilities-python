from copy import deepcopy

def get_items_from_list_conditionally(list_, condition_function, whether_deepcopy = False):
    """ This is reference-copy function not value-copy.
    
    Examples
    --------
    test_list = [{'a':1, 'b':2}, {'a':3, 'b':2}, {'a':2, 'b':4}]\n
    print(get_items_from_list_conditionally(test_list, lambda x: True if x['a'] >= 2 else False))
        == [{'a': 3, 'b': 2}, {'a': 2, 'b': 4}]
    """

    assert(type(list_) == type([]))
    result_collection = []
    for item in list_:
        if condition_function(item):
            result_collection.append(deepcopy(item)) if whether_deepcopy else result_collection.append(item)
    return result_collection

def merge_dictionaries(list_of_dicts : list, use_last_when_overlapped = True):
    """ Merge dictionaries with 'update' method.
        Always deepcopy because of nature of dict.update method.

    Examples
    --------
    test_list = [{'a':1, 'b':2}, {'a':3, 'b':2}, {'a':2, 'b':4}]\n
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
        if list_of_dicts[dict_idx] is not None:
            assert(type(list_of_dicts[dict_idx]) == type({}))
            merged_dict.update(list_of_dicts[dict_idx])
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

if __name__ == "__main__":
    test_dict = {"n": ["a", "b", "c"], "m": ["d", "e", "f", "g"], 1: ["h", "i"]}
    print(copy_dict_and_delete_element(dict_to_copy = test_dict, list_of_keys_to_delete = ["n", 1]))