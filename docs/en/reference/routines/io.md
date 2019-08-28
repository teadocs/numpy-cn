# Input and output

## NumPy binary files (NPY, NPZ)

method | description
---|---
[load](generated/numpy.load.html#numpy.load)(file[, mmap_mode, allow_pickle, …]) | Load arrays or pickled objects from .npy, .npz or pickled files.
[save](generated/numpy.save.html#numpy.save)(file, arr[, allow_pickle, fix_imports]) | Save an array to a binary file in NumPy .npy format.
[savez](generated/numpy.savez.html#numpy.savez)(file, \*args, \*\*kwds) | Save several arrays into a single file in uncompressed .npz format.
[savez_compressed](generated/numpy.savez_compressed.html#numpy.savez_compressed)(file, \*args, \*\*kwds) | Save several arrays into a single file in compressed .npz format.

The format of these binary file types is documented in
[``numpy.lib.format``](generated/numpy.lib.format.html#module-numpy.lib.format)

## Text files

method | description
---|---
[loadtxt](generated/numpy.loadtxt.html#numpy.loadtxt)(fname[, dtype, comments, delimiter, …]) | Load data from a text file.
[savetxt](generated/numpy.savetxt.html#numpy.savetxt)(fname, X[, fmt, delimiter, newline, …]) | Save an array to a text file.
[genfromtxt](generated/numpy.genfromtxt.html#numpy.genfromtxt)(fname[, dtype, comments, …]) | Load data from a text file, with missing values handled as specified.
[fromregex](generated/numpy.fromregex.html#numpy.fromregex)(file, regexp, dtype[, encoding]) | Construct an array from a text file, using regular expression parsing.
[fromstring](generated/numpy.fromstring.html#numpy.fromstring)(string[, dtype, count, sep]) | A new 1-D array initialized from text data in a string.
[ndarray.tofile](generated/numpy.ndarray.tofile.html#numpy.ndarray.tofile)(fid[, sep, format]) | Write array to a file as text or binary (default).
[ndarray.tolist](generated/numpy.ndarray.tolist.html#numpy.ndarray.tolist)() | Return the array as an a.ndim-levels deep nested list of Python scalars.

## Raw binary files

method | description
---|---
[fromfile](generated/numpy.fromfile.html#numpy.fromfile)(file[, dtype, count, sep, offset]) | Construct an array from data in a text or binary file.
[ndarray.tofile](generated/numpy.ndarray.tofile.html#numpy.ndarray.tofile)(fid[, sep, format]) | Write array to a file as text or binary (default).

## String formatting

method | description
---|---
[array2string](generated/numpy.array2string.html#numpy.array2string)(a[, max_line_width, precision, …]) | Return a string representation of an array.
[array_repr](generated/numpy.array_repr.html#numpy.array_repr)(arr[, max_line_width, precision, …]) | Return the string representation of an array.
[array_str](generated/numpy.array_str.html#numpy.array_str)(a[, max_line_width, precision, …]) | Return a string representation of the data in an array.
[format_float_positional](generated/numpy.format_float_positional.html#numpy.format_float_positional)(x[, precision, …]) | Format a floating-point scalar as a decimal string in positional notation.
[format_float_scientific](generated/numpy.format_float_scientific.html#numpy.format_float_scientific)(x[, precision, …]) | Format a floating-point scalar as a decimal string in scientific notation.

## Memory mapping files

method | description
---|---
[memmap](generated/numpy.memmap.html#numpy.memmap) | Create a memory-map to an array stored in a binary file on disk.

## Text formatting options

method | description
---|---
[set_printoptions](generated/numpy.set_printoptions.html#numpy.set_printoptions)([precision, threshold, …]) | Set printing options.
[get_printoptions](generated/numpy.get_printoptions.html#numpy.get_printoptions)() | Return the current print options.
[set_string_function](generated/numpy.set_string_function.html#numpy.set_string_function)(f[, repr]) | Set a Python function to be used when pretty printing arrays.
[printoptions](generated/numpy.printoptions.html#numpy.printoptions)(\\\*args, \\\*\\\*kwargs) | Context manager for setting print options.

## Base-n representations

method | description
---|---
[binary_repr](generated/numpy.binary_repr.html#numpy.binary_repr)(num[, width]) | Return the binary representation of the input number as a string.
[base_repr](generated/numpy.base_repr.html#numpy.base_repr)(number[, base, padding]) | Return a string representation of a number in the given base system.

## Data sources

method | description
---|---
[DataSource](generated/numpy.DataSource.html#numpy.DataSource)([destpath]) | A generic data source file (file, http, ftp, …).

## Binary Format Description

method | description
---|---
[lib.format](generated/numpy.lib.format.html#module-numpy.lib.format) | Binary serialization