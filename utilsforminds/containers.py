def get_items_from_list_conditionally(list_, condition_function):
    """ This is reference-copy function not value-copy.
    
    Examples
    --------
    test_list = [{'a':1, 'b':2}, {'a':3, 'b':2}, {'a':2, 'b':4}]
    print(get_items_from_list_conditionally(test_list, lambda x: True if x['a'] >= 2 else False))
        -\> [{'a': 3, 'b': 2}, {'a': 2, 'b': 4}]
    """

    assert(type(list_) == type([]))
    result_collection = []
    for item in list_:
        if condition_function(item):
            result_collection.append(item)
    return result_collection

if __name__ == "__main__":
    test_list = [{'a':1, 'b':2}, {'a':3, 'b':2}, {'a':2, 'b':4}]
    print(get_items_from_list_conditionally(test_list, lambda x: True if x['a'] >= 2 else False))