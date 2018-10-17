#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 10:57:55 2018

@author: brendan
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from constraints import Constraint
import config


def main():
    """main file to execute the sampler
    """
    
    # read the user's input arguments
    input_file, output_file, n_results = read_input()
    
    # create a constraints object
    hyperspace = get_constraints(input_file)
    
    # extract the example point and use it to initialize the array of
    # acceptable points
    example = hyperspace.get_example()
    points_init = make_points_array(example, n_results)
    
    # run the sampler
    max_steps = config.MAX_STEPS
    step_size = config.INIT_STEP_SIZE
    points = run_sampler(hyperspace, points_init, step_size, max_steps)
    
    plt.plot(points[:,0], points[:,1], marker="o", linestyle="")
    plt.show()
    
    # write the output to a file
    write_output(points, output_file)
    


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
    
    if n < 1:
        print("Error: n_results is negative -- 1 output will be produced")
        n = 1
    
    points = np.tile(vec, [n, 1])
    return points


def write_output(array, output_path):
    """writes a two-dimensional array to an output file given by output_path
    """
    
    # split output_path into a base path and a file name
    base, fname = os.path.split(output_path)
    
    # if the user specified a bsae path but that directory does not exist,
    # then create that directory
    if base and not os.path.isdir(base):
        os.makedirs(base)
    
    # if the user did not specify a valid file name (just a directory),
    # then throw an error
    if not fname:
        sys.exit("Error: no output file specified")
    
    # save the array to output_path with 4 digits of precision,
    # spaces between coordinates, and new lines between each vector
    np.savetxt(output_path, array, fmt='%.4f', delimiter=' ', newline='\n')


def run_sampler(hyperspace, points, step_size, max_steps):
    """runs Markov Chain Monte Carlo to sample some space
        hyperspace is an instance of the Constraints class
        points is an array of acceptable points
        step_size is a measure of how large the step sizes are
        max_steps is the maximum number of steps that can be run
    """
    
    ndims = hyperspace.get_ndim() # number of dimensions
    
    # several arrays for tracking the sampler
    # pos_mean and pos_std are two-dimensional arrays tracking the average
    # and standard deviation of position along each coordinate axis
    pos_mean = np.zeros((max_steps, ndims))
    pos_std = np.zeros((max_steps, ndims))
    # accept_rate is a one-dimensional array tracking the acceptance raate
    # for each step
    accept_rate = np.zeros(max_steps)
    
    # loop for up to the maximum number of steps
    for i in range(max_steps):
        # run a single step
        points, accept_rate[i] = run_step(points, hyperspace, step_size)
        # calculate the average and std of position after this step
        pos_mean[i] = calc_pos_mean(points)
        pos_std[i] = calc_std_mean(points)
        
        # decide whether or not it's time to check in on the sampler and
        # see if it's ready to stop or the step size should be modified
        if evaluate_sampler(i):
            # check if the sampler has stabilized
            if sampler_stable(i, pos_mean, pos_std):
                # if it has stabilized, break the loop
                print("Sampler stopped after {0} steps".format(i))
                break
            # possibly modify the step size
            step_size = modify_step_size(step_size, accept_rate)
        
    if i == max_steps-1:
        print("""Warning: reached maximum number of steps. Sampler may not be converged""")
    
    plt.plot(range(max_steps), pos_mean[:,0])
    plt.plot(range(max_steps), pos_mean[:,1])
    plt.plot(range(max_steps), pos_std[:,0])
    plt.plot(range(max_steps), pos_std[:,1])
    plt.show()
    
    return points


def run_step(points, hyperspace, step_size):
    """runs a single step of the Markov chains
        points is an array of current points in the space
        hyperspace is the object containing the list of constraints
        step_size is a float indicating roughly how large the steps are
        returns both the new array of points and the acceptance rate
    """
    
    # number of points and number of dimensions for each point
    npoints, ndims = points.shape
    
    valid_moves = 0 # keeps track of how many valid moves there have been
    
    # loop through the points
    for i in range(npoints):
        # calculate a step. It is an ndims dimensional Gaussian of mean 0
        # and standard deviation step_size
        step = np.random.normal(scale=step_size, size=ndims)
        # add step to the current position, and then take each coordinate
        # modulo 1 to remain in the n-dimensional hypercube
        new_vec = np.mod(points[i] + step, 1)
        
        # if this new vector meets all of the constraints, then modify the
        # points array and iterate valid_moves
        if hyperspace.apply(new_vec):
            points[i] = new_vec
            valid_moves = valid_moves + 1
    
    acceptance_rate = valid_moves/npoints # calculate the acceptance rate
    
    return points, acceptance_rate


def calc_pos_mean(points):
    """calculate the average position of a set of vectors stored in the
        two-dimensional array, points
    """
    return np.mean(points, axis=0)


def calc_std_mean(points):
    """calculate the standard deviation of a set of vectors stored in the
        two-dimensional array, points
    """
    return np.std(points, axis=0)

def evaluate_sampler(i):
    """decides whether or not its time to evaluate the sampler
        This is done in a simple manner -- every fixed number of steps,
        given by config.CHECK_STEPS, the sampler is evaluated
    """
    check_steps = config.CHECK_STEPS
    return np.mod(i, check_steps)


def sampler_stable(i, pos_mean, pos_std):
    """decides whether or not the sampler has stabilized
    """
    return False


def modify_step_size(step_size, accept_rate):
    """
    """
    
    return step_size
            

if __name__ == '__main__':
    main()