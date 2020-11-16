import os
import numbers

def format_extension(path : str, format : str):
    """Format extension

    Examples
    --------
        print(format_extension("/path/to/somefile.txt", "tex"))
        >>>/path/to/somefile.tex
    
    """
    file_name, file_extension = os.path.splitext(path)
    return f"{file_name}.{format}"

# def is_printable_object(something):
#     if isinstance(something, (numbers.Number, type('a'), type(True), type(None))):
#         return True
#     else:
#         return False

# def paramDictToStr(param_dict, num_prints_limit = 20, level = 0):
#     if is_printable_object(param_dict): ## Base Case: This is itself string or number.
#         return str(param_dict)
#     elif isinstance(param_dict, (list, tuple, type(range(3)))): ## Recursive Case : list
#         resulted_str = "["
#         for idx in range(min(len(param_dict), num_prints_limit)):
#             resulted_str = resulted_str + paramDictToStr(param_dict[idx], num_prints_limit = num_prints_limit) + ", "
#         return resulted_str + "]"
#     elif isinstance(param_dict, dict): ## Recursive Case : dict
#         resulted_str = "\t" * level + "{\n"
#         for key, num_printed in zip(param_dict.keys(), range(len(param_dict))):
#             if num_printed >= num_prints_limit:
#                 break
#             resulted_str = resulted_str + "\t" * (level + 1) + f"{key}: {paramDictToStr(param_dict[key], num_prints_limit = num_prints_limit, level = level + 1)},\n"
#             num_printed += 1
#         return "\t" * level + resulted_str + "}\n"
#     # elif hasattr(param_dict, '__dict__'):
#     #     return paramDictToStr(param_dict.__dict__, num_prints_limit)
#     #     a.property
#     else:
#         return "Non-printable"

class Formatter(object):
    """Container -> string Formatter.
    
    ref: https://stackoverflow.com/questions/3229419/how-to-pretty-print-nested-dictionaries
    """
    def __init__(self, whether_print_object = False, limit_number_of_prints_in_container = 20, recursive_prints = True):
        self.types = {}
        self.htchar = '\t'
        self.lfchar = '\n'
        self.indent = 0
        self.set_formater(object, self.__class__.format_object)
        self.set_formater(dict, self.__class__.format_dict)
        self.set_formater(list, self.__class__.format_list)
        self.set_formater(tuple, self.__class__.format_tuple)
        self.whether_print_object = whether_print_object
        self.limit_number_of_prints_in_container = limit_number_of_prints_in_container
        self.recursive_prints = recursive_prints

    def set_formater(self, obj, callback):
        self.types[obj] = callback

    def cut_container_to_limit(self, value):
        """Cut the size of container.
        
        Parameters
        ----------
        value : object
            Something to print, possibly string, list, tuple, dictionary, number, .., every object.
        """

        if isinstance(value, dict):
            value_shrink_in_limit = {}
            count = 0
            for key in value.keys():
                if count >= self.limit_number_of_prints_in_container:
                    break
                if self.recursive_prints or not isinstance(value[key], (list, tuple, dict, type(range(7)))):
                    value_shrink_in_limit[key] = value[key]
                    count += 1
        elif isinstance(value, (list, tuple, type(range(7)))):
            value_shrink_in_limit = []
            for idx in range(min(self.limit_number_of_prints_in_container, len(value))):
                if self.recursive_prints or not isinstance(value[idx], (list, tuple, dict, type(range(7)))):
                    value_shrink_in_limit.append(value[idx])
        else:
            value_shrink_in_limit = value
        return value_shrink_in_limit

    def __call__(self, value, **args):
        for key in args:
            setattr(self, key, args[key])
        formater = self.types[type(value) if type(value) in self.types else object]
        return formater(self, value, self.indent)

    def format_object(self, value, indent):
        if self.whether_print_object or isinstance(value, (numbers.Number, str)):
            return repr(value)
        else:
            return "Non-printable"
        # return repr(value)

    def format_dict(self, value, indent):
        ## Cut the size of container.
        value_shrink_in_limit = self.cut_container_to_limit(value)

        items = [
            self.lfchar + self.htchar * (indent + 1) + repr(key) + ': ' +
            (self.types[type(value_shrink_in_limit[key]) if type(value_shrink_in_limit[key]) in self.types else object])(self, value_shrink_in_limit[key], indent + 1)
            for key in value_shrink_in_limit
        ]
        return '{%s}' % (','.join(items) + self.lfchar + self.htchar * indent)

    def format_list(self, value, indent):
        ## Cut the size of container.
        value_shrink_in_limit = self.cut_container_to_limit(value)

        items = [
            self.lfchar + self.htchar * (indent + 1) + (self.types[type(item) if type(item) in self.types else object])(self, item, indent + 1)
            for item in value_shrink_in_limit
        ]
        return '[%s]' % (','.join(items) + self.lfchar + self.htchar * indent)

    def format_tuple(self, value, indent):
        ## Cut the size of container.
        value_shrink_in_limit = self.cut_container_to_limit(value)

        items = [
            self.lfchar + self.htchar * (indent + 1) + (self.types[type(item) if type(item) in self.types else object])(self, item, indent + 1)
            for item in value_shrink_in_limit
        ]
        return '(%s)' % (','.join(items) + self.lfchar + self.htchar * indent)

def container_to_str(container, whether_print_object = False, limit_number_of_prints_in_container = 20, recursive_prints = True):
    """Pretty print for parameter print.
    
    ref: https://stackoverflow.com/questions/3229419/how-to-pretty-print-nested-dictionaries

    Parameters
    ----------
    whether_print_object : bool
        Whether to print special object, which is not container/number/string.
    limit_number_of_prints_in_container : int
        Maximum number of elements to print in a container.
    recursive_prints : bool
        Whether to print the container in a recursive containers. 

    Examples
    --------
    test_dict = {'hi':[1, {'hello': [3, 4], "die": "live"}], 'end': [3, 6], 7: "hihi", 8: "hellobye", 10: 11, 12: 13}\n
    print(container_to_str(test_dict))
        >>> {
        'hi': [
                1,
                {
                        'hello': [
                                3,
                                4
                        ],
                        'die': 'live'
                }
        ],
        'end': [
                3,
                6
        ],
        7: 'hihi',
        8: 'hellobye',
        10: 11,
        12: 13
        }
    """

    return Formatter(whether_print_object = whether_print_object, limit_number_of_prints_in_container = limit_number_of_prints_in_container, recursive_prints = recursive_prints)(container)

if __name__ == "__main__":
    test_dict = {'hi':[1, {'hello': [3, [7 for i in range(100)]], "die": "live"}], 'end': [3, 6], 7: "hihi", 8: "hellobye", 10: 11, 12: 13, 22: (lambda x: True)}
    print(container_to_str(test_dict))
    