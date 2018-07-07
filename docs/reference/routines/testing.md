# 测试相关API

Common test support for all numpy test scripts.

This single module should provide all the common functionality for numpy tests in a single location, so that test scripts can just import it and work right away.

## Asserts

- assert_almost_equal(actual, desired[, …])	Raises an AssertionError if two items are not equal up to desired precision.
- assert_approx_equal(actual, desired[, …])	Raises an AssertionError if two items are not equal up to significant digits.
- assert_array_almost_equal(x, y[, decimal, …])	Raises an AssertionError if two objects are not equal up to desired precision.
- assert_allclose(actual, desired[, rtol, …])	Raises an AssertionError if two objects are not equal up to desired tolerance.
- assert_array_almost_equal_nulp(x, y[, nulp])	Compare two arrays relatively to their spacing.
- assert_array_max_ulp(a, b[, maxulp, dtype])	Check that all items of arrays differ in at most N Units in the Last Place.
- assert_array_equal(x, y[, err_msg, verbose])	Raises an AssertionError if two array_like objects are not equal.
- assert_array_less(x, y[, err_msg, verbose])	Raises an AssertionError if two array_like objects are not ordered by less than.
- assert_equal(actual, desired[, err_msg, verbose])	Raises an AssertionError if two objects are not equal.
- assert_raises(exception_class, callable, …)	Fail unless an exception of class exception_class is thrown by callable when invoked with arguments args and keyword arguments kwargs.
- assert_raises_regex(exception_class, …)	Fail unless an exception of class exception_class and with message that matches expected_regexp is thrown by callable when invoked with arguments args and keyword arguments kwargs.
- assert_warns(warning_class, *args, **kwargs)	Fail unless the given callable throws the specified warning.
- assert_string_equal(actual, desired)	Test if two strings are equal.

## Decorators

- decorators.deprecated([conditional])	Filter deprecation warnings while running the test suite.
- decorators.knownfailureif(fail_condition[, msg])	Make function raise KnownFailureException exception if given condition is true.
- decorators.setastest([tf])	Signals to nose that this function is or is not a test.
- decorators.skipif(skip_condition[, msg])	Make function raise SkipTest exception if a given condition is true.
- decorators.slow(t)	Label a test as ‘slow’.
- decorate_methods(cls, decorator[, testmatch])	Apply a decorator to all methods in a class matching a regular expression.

## Test Running

- Tester	alias of numpy.testing.nose_tools.nosetester.NoseTester
- run_module_suite([file_to_run, argv])	Run a test module.
- rundocs([filename, raise_on_error])	Run doctests found in the given file.
- suppress_warnings([forwarding_rule])	Context manager and decorator doing much the same as warnings.catch_warnings.