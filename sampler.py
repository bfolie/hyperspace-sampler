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

    input_file = sys.argv[1]  # first argument is the input file name
    output_file = sys.argv[2]  # second argument is the output file name

    # third argument is the number of results to output,
    # which must be an integer. If it is not an integer, throw an error
    try:
        n_results = int(sys.argv[3])  # convert the input string to an int
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

    # stack vec on top of itself n times
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

    ndims = hyperspace.get_ndim()  # number of dimensions

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
            # possibly modify the step size
            step_size, step_mod = modify_step_size(i, step_size, accept_rate)

            # if the step size was not modified, then check if the sampler
            # has stabilized (if step size was modified then it's probably)
            # not stable
            if sampler_stable(i, pos_mean, pos_std) and not step_mod:
                # if it has stabilized, break the loop
                print("Sampler stopped after {0} steps".format(i))
                break

    if i == max_steps-1:
        print("""Warning: reached maximum number of steps. Sampler may not be converged""")

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

    valid_moves = 0  # keeps track of how many valid moves there have been

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

    acceptance_rate = valid_moves/npoints  # calculate the acceptance rate

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
    return np.mod(i, check_steps) == 0 and i > 0


def sampler_stable(index, pos_mean, pos_std):
    """decides whether or not the sampler has stabilized
        index is the current index
        pos_mean and pos_std are two-dimensional arrays holding the mean and
        standard deviation of each coordinate after each step
    """

    check_steps = config.CHECK_STEPS
    # number of points and number of dimensions for each point
    nsteps, ndims = pos_mean.shape

    # cycle through the columns of pos_mean, each of which corresponds
    # to a different coordinate
    for coordinate_mean in pos_mean.T:
        # check to see if this array has stabilized, and if not, return False
        # (i.e. if one coordinate is not stable, the whole simulation
        # is not stable)
        if not array_stable(index, check_steps, coordinate_mean):
            return False

    # same as above but for std_mean
    for coordinate_std in pos_std.T:
        if not array_stable(index, check_steps, coordinate_std):
            return False
    
    return True


def array_stable(index, num_steps, A):
    """checks to see if array A has stabilized
        index is the current index of the array
        num_steps is how many steps back in the past we consider
        Compare from index-num_steps to index and from index-2*num_steps to
        index-num_steps, seeing if there is a substantial difference
    """
    
    # if index is not at least num_steps*2, then we don't have enough data
    # points to check, and if index if greater than len(A)-1, then something
    # has gone wrong
    if index < 2*num_steps or index > len(A)-1:
        return False
    
    # pull out the most recent num_steps steps (A1) and also the num_steps
    # before that
    A1 = A[index-num_steps+1:index+1]
    A2 = A[index-2*num_steps+1:index-num_steps+1]
    
    # calculate the mean and standard deviations of these two arrays
    val1 = np.mean(A1)
    val2 = np.mean(A2)
    std1 = np.std(A1)
    std2 = np.std(A2)
    
    # if val1 and val2 differ by at most config.TOLERANCE, then array
    # is possibly stabilized
    check1 = np.abs(val1 - val2) < config.TOLERANCE
    
    # if val1 and val2 are statistically indistinguishable to one sigma,
    # then the array is possibly stabilized
    check2 = np.abs(val1 - val2) < 1*np.sqrt(std1**2 + std2**2)
    
    return check1 and check2
    


def modify_step_size(index, step_size, accept_rate):
    """decides whether or not to modify the step size
        If steps are accepted too frequently, make the step size bigger
        If they are rejected too frequently, make the step size smaller
        accept_rate is an array containing the acceptance rate at each step
        step_size is the current step size
        index is the index of the most recent step
        Returns both the new step size and True/False depending on whether
        or not step_size was modified
    """
    check_steps = config.CHECK_STEPS

    # make sure enough steps have passed
    if index < check_steps:
        return step_size, False

    # take the most recent check_steps steps and average the acceptance rate
    recent_accept_rate = np.mean(accept_rate[index-check_steps+1:index+1])

    # import constants -- minimum acceptance rate, maximum acceptance rate,
    # and how much to change the step size by if the acceptance is too
    # large or too small
    min_accept_rate = config.MIN_ACCEPT_RATE
    max_accept_rate = config.MAX_ACCEPT_RATE
    factor = config.STEP_SIZE_FACTOR

    # make sure factor is between 0 and 1, and if not just set it to 1
    # (in which case the step size won't change)
    if factor <= 0 or factor > 1:
        factor = 1

    # if the recent acceptance rate has been below the minimum acceptance
    # rate, then make the steps smaller
    if recent_accept_rate < min_accept_rate:
        print("Step size made smaller")
        return step_size*factor, True

    # if the recent acceptance rate has been above the maximum acceptance
    # rate, then make the steps bigger (but not bigger than 1)
    if recent_accept_rate > max_accept_rate:
        if step_size/factor < 1:
            print("Step size made bigger")
            return step_size/factor, True

    # otherwise the recent acceptance rate was in an OK range, so step_size
    # does not change
    return step_size, False


if __name__ == '__main__':
    main()
