#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 10:57:55 2018

@author: brendan
"""

import sys
import numpy as np
from constraints import Constraint

def main():
    """main file to execute the sampler
    """
    input_file, output_file, n_results = read_input()
    
    space = get_constraints(input_file)
    
    example = space.get_example()
    ndims = space.get_ndim
    
    points = make_points_array(example, n_results)
    


def read_input():
    """reads the command line input and checks for errors
    """
    
    # make sure that there are at least 3 arguments supplied
    # sys.argv[0] is the name of this file (sampler.py), whereas sys.argv[1],
    # [2], and [3] are the arguments of interest.
    # Any additional arguments are ignored
    n_args = len(sys.argv)
    if n_args < 4:
        sys.exit("Error: not enough arguments supplied")
    
    input_file = sys.argv[1] # first argument is the input file name
    output_file = sys.argv[2] # second argument is the output file name
    
    # third argument is the number of results to output,
    # which must be an integer. If it is not an integer, throw an error
    try:
        n_results = int(sys.argv[3]) # convert the input string to an int
    except ValueError:
        sys.exit("Error: third input (number of results) must be an integer")
        
    
    return input_file, output_file, n_results


def get_constraints(input_file):
    """reads the input file and creates an instance of the Constraint class
    """
    
    # create a new instance of the Constraint class by reading input_file
    # catch several errors: file not found, first two lines (number of
    # dimensions and starting vector) not formatted correctly, or syntax
    # error in the constraints
    try:
        space = Constraint(input_file)
    except FileNotFoundError:
        sys.exit("Error: input file not found")
    except ValueError:
        sys.exit("Error: input file formatted improperly")
    except SyntaxError:
        sys.exit("Error: syntax error in constraints")
    
    # get the example point and make sure it has the correct dimensionality
    example = space.get_example()
    if len(example) < space.get_ndim():
        sys.exit("Error: example point does not have enough dimensions")
    
    # check and make sure the example point actually satisfies all of the
    # constraints. This additionally serves to make sure the constraints
    # are specified correctly
    try:
        check_example = space.apply(example)
    except IndexError:
        sys.exit("Error: invalid constraints")
    except NameError:
        sys.exit("Error: invalid constraints")
    
    # if space.apply(example) returned false, then throw an error
    if not check_example:
        sys.exit("Error: example point is invalid")
    
    return space


def make_points_array(vec, n):
    """creates an array of all valid points in the space
        Essentially just takes the example vector, vec, and copies it n times
        The output array has size n by len(vec)
    """
            

if __name__ == '__main__':
    main()