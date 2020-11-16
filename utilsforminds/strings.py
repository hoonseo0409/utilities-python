import os

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
    def __init__(self):
        self.types = {}
        self.htchar = '\t'
        self.lfchar = '\n'
        self.indent = 0
        self.set_formater(object, self.__class__.format_object)
        self.set_formater(dict, self.__class__.format_dict)
        self.set_formater(list, self.__class__.format_list)
        self.set_formater(tuple, self.__class__.format_tuple)

    def set_formater(self, obj, callback):
        self.types[obj] = callback

    def __call__(self, value, **args):
        for key in args:
            setattr(self, key, args[key])
        formater = self.types[type(value) if type(value) in self.types else object]
        return formater(self, value, self.indent)

    def format_object(self, value, indent):
        return repr(value)

    def format_dict(self, value, indent):
        items = [
            self.lfchar + self.htchar * (indent + 1) + repr(key) + ': ' +
            (self.types[type(value[key]) if type(value[key]) in self.types else object])(self, value[key], indent + 1)
            for key in value
        ]
        return '{%s}' % (','.join(items) + self.lfchar + self.htchar * indent)

    def format_list(self, value, indent):
        items = [
            self.lfchar + self.htchar * (indent + 1) + (self.types[type(item) if type(item) in self.types else object])(self, item, indent + 1)
            for item in value
        ]
        return '[%s]' % (','.join(items) + self.lfchar + self.htchar * indent)

    def format_tuple(self, value, indent):
        items = [
            self.lfchar + self.htchar * (indent + 1) + (self.types[type(item) if type(item) in self.types else object])(self, item, indent + 1)
            for item in value
        ]
        return '(%s)' % (','.join(items) + self.lfchar + self.htchar * indent)

def container_to_str(container):
    """Pretty print for parameter print.
    
    ref: https://stackoverflow.com/questions/3229419/how-to-pretty-print-nested-dictionaries

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
    
    return Formatter()(container)

if __name__ == "__main__":
    test_dict = {'hi':[1, {'hello': [3, 4], "die": "live"}], 'end': [3, 6], 7: "hihi", 8: "hellobye", 10: 11, 12: 13}
    print(container_to_str(test_dict))