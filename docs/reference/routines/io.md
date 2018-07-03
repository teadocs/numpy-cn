# 输入和输出

## NumPy binary files (NPY, NPZ)

- load(file[, mmap_mode, allow_pickle, …])	Load arrays or pickled objects from .npy, .npz or pickled files.
- save(file, arr[, allow_pickle, fix_imports])	Save an array to a binary file in NumPy .npy format.
- savez(file, *args, **kwds)	Save several arrays into a single file in uncompressed .npz format.
- savez_compressed(file, *args, **kwds)	Save several arrays into a single file in compressed .npz format.

The format of these binary file types is documented in [http://docs.scipy.org/doc/numpy/neps/npy-format.html](http://docs.scipy.org/doc/numpy/neps/npy-format.html)

## Text files

- loadtxt(fname[, dtype, comments, delimiter, …])	Load data from a text file.
- savetxt(fname, X[, fmt, delimiter, newline, …])	Save an array to a text file.
- genfromtxt(fname[, dtype, comments, …])	Load data from a text file, with missing values handled as specified.
- fromregex(file, regexp, dtype[, encoding])	Construct an array from a text file, using 

## regular expression parsing.

- fromstring(string[, dtype, count, sep])	A new 1-D array initialized from text data in a string.
- ndarray.tofile(fid[, sep, format])	Write array to a file as text or binary (default).
- ndarray.tolist()	Return the array as a (possibly nested) list.

## Raw binary files

- fromfile(file[, dtype, count, sep])	Construct an array from data in a text or binary file.
- ndarray.tofile(fid[, sep, format])	Write array to a file as text or binary (default).

## String formatting

- array2string(a[, max_line_width, precision, …])	Return a string representation of an array.
- array_repr(arr[, max_line_width, precision, …])	Return the string representation of an array.
- array_str(a[, max_line_width, precision, …])	Return a string representation of the data in an array.
- format_float_positional(x[, precision, …])	Format a floating-point scalar as a decimal string in positional notation.
- format_float_scientific(x[, precision, …])	Format a floating-point scalar as a decimal string in scientific notation.

## Memory mapping files

- memmap	Create a memory-map to an array stored in a binary file on disk.

## Text formatting options

- set_printoptions([precision, threshold, …])	Set printing options.
- get_printoptions()	Return the current print options.
- set_string_function(f[, repr])	Set a Python function to be used when pretty printing arrays.

## Base-n representations

- binary_repr(num[, width])	Return the binary representation of the input number as a string.
- base_repr(number[, base, padding])	Return a string representation of a number in the given base system.

## Data sources

- DataSource([destpath])	A generic data source file (file, http, ftp, …).