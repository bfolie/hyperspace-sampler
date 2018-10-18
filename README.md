Randomly samples a high-dimensional space with a set of constraints.
Constraints file must have the following format:
    first line is the number of dimensions, ndims
    second line is a list of ndims numbers specifying a point at which the constraints are all met
    third line is an optional comment line (begins with #)
    beyond that each line is an equation giving a constraint, such as
    x[0] + x[1]**2/x[3] - 0.1 >= 0.0

To install, after unapcking the tar.gz distribution navigate to the directory containing setup.py and run the following command:
    python3 setup.py install

Run the application with the following command:
python3 sampler.py <input_file> <output_file> <n_results>

<input_file> is the path to a file that contains a description of the space
and constraints.
<output_file> is a path to where the output should be written
<n_results> is the number of valid points that should be output