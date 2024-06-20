utils
===============================

version number: 0.0.1
author: Hoon Seo

Overview
--------

This repository contains a collection of Python utilities used across various research projects. These utilities provide various visualization functions and handle common data structures in machine learning research, such as Numpy arrays, TensorFlow tensors, dictionaries, and lists. Additionally, it offers a flexible way to automate grid search for hyperparameter tuning of machine learning models. These utility functions are designed to facilitate research and avoid redundant coding across multiple projects.

Project Description
-------------------
`biomarkers.py`: Processes and visualizes neuroimaging and genomic data.
`containers.py`: Handles Python native data structures such as dictionaries and lists. For example, it generates grids from hyperparameters represented by dictionaries of lists.
`decorators.py`: Provides [Python Decorator](https://peps.python.org/pep-0318/) to add additional features to existing functions. For example, a decorator to detect abnormal (NaN) values in function inputs or to redirect the function to another function with the same arguments.
`helpers.py`: Contains miscellaneous functions primarily used within this package. For example, it handles paths and generates files/folders.
`math.py`: Involves the calculation of norms, loss, regularization, and statistics.
`numpy_array.py`: Provides utility functions to manipulate Numpy arrays.
`strings.py`: Provides a class for pretty formatting of strings.
`tensors.py`: Offers utility functions for TensorFlow tensors, such as parallelizing for loops.
`visualization.py`: Contains various visualization functions, including graphs, bar charts, 2D/3D points, and alpha-shapes.

Installation / Usage
--------------------

To install use pip:

    $ pip install -e path_to_utilsforminds

Or use pipenv:

    $ pipenv install -e path_to_utilsforminds

Or clone the repo:

    $ git clone https://gitlab.com/hoonseo/utilsforminds.git
    $ python setup.py install
    
Contributing
------------

We welcome contributions to improve the framework and extend its functionalities. Please feel free to fork the repository, make your changes, and submit a pull request.

