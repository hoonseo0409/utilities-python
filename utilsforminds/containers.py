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
        assert(type(list_of_dicts[dict_idx]) == type({}))
        merged_dict.update(list_of_dicts[dict_idx])
    return merged_dict

if __name__ == "__main__":
    test_list = [{'a':1, 'b':2}, {'a':3, 'b':2}, {'a':2, 'b':4, 'c':9}]
    print(get_items_from_list_conditionally(test_list, lambda x: True if x['a'] >= 2 else False))

    print(merge_dictionaries(test_list, use_last_when_overlapped = False))