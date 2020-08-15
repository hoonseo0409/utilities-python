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

if __name__ == "__main__":
    print(format_extension("/path/to/somefile.txt", "tex"))