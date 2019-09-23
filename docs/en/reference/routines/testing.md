# Test Support (``numpy.testing``)

Common test support for all numpy test scripts.

This single module should provide all the common functionality for numpy
tests in a single location, so that [test scripts](https://numpy.org/devdocs/dev/development_environment.html#development-environment) can just import it and work right away. For
background, see the [Testing Guidelines](testing.html#testing-guidelines)

## Asserts

method | description
---|---
[assert_almost_equal](https://numpy.org/devdocs/reference/generated/numpy.testing.assert_almost_equal.html#numpy.testing.assert_almost_equal)(actual, desired[, …]) | Raises an AssertionError if two items are not equal up to desired precision.
[assert_approx_equal](https://numpy.org/devdocs/reference/generated/numpy.testing.assert_approx_equal.html#numpy.testing.assert_approx_equal)(actual, desired[, …]) | Raises an AssertionError if two items are not equal up to significant digits.
[assert_array_almost_equal](https://numpy.org/devdocs/reference/generated/numpy.testing.assert_array_almost_equal.html#numpy.testing.assert_array_almost_equal)(x, y[, decimal, …]) | Raises an AssertionError if two objects are not equal up to desired precision.
[assert_allclose](https://numpy.org/devdocs/reference/generated/numpy.testing.assert_allclose.html#numpy.testing.assert_allclose)(actual, desired[, rtol, …]) | Raises an AssertionError if two objects are not equal up to desired tolerance.
[assert_array_almost_equal_nulp](https://numpy.org/devdocs/reference/generated/numpy.testing.assert_array_almost_equal_nulp.html#numpy.testing.assert_array_almost_equal_nulp)(x, y[, nulp]) | Compare two arrays relatively to their spacing.
[assert_array_max_ulp](https://numpy.org/devdocs/reference/generated/numpy.testing.assert_array_max_ulp.html#numpy.testing.assert_array_max_ulp)(a, b[, maxulp, dtype]) | Check that all items of arrays differ in at most N Units in the Last Place.
[assert_array_equal](https://numpy.org/devdocs/reference/generated/numpy.testing.assert_array_equal.html#numpy.testing.assert_array_equal)(x, y[, err_msg, verbose]) | Raises an AssertionError if two array_like objects are not equal.
[assert_array_less](https://numpy.org/devdocs/reference/generated/numpy.testing.assert_array_less.html#numpy.testing.assert_array_less)(x, y[, err_msg, verbose]) | Raises an AssertionError if two array_like objects are not ordered by less than.
[assert_equal](https://numpy.org/devdocs/reference/generated/numpy.testing.assert_equal.html#numpy.testing.assert_equal)(actual, desired[, err_msg, verbose]) | Raises an AssertionError if two objects are not equal.
[assert_raises](https://numpy.org/devdocs/reference/generated/numpy.testing.assert_raises.html#numpy.testing.assert_raises)(exception_class, callable, …) | Fail unless an exception of class exception_class is thrown by callable when invoked with arguments args and keyword arguments kwargs.
[assert_raises_regex](https://numpy.org/devdocs/reference/generated/numpy.testing.assert_raises_regex.html#numpy.testing.assert_raises_regex)(exception_class, …) | Fail unless an exception of class exception_class and with message that matches expected_regexp is thrown by callable when invoked with arguments args and keyword arguments kwargs.
[assert_warns](https://numpy.org/devdocs/reference/generated/numpy.testing.assert_warns.html#numpy.testing.assert_warns)(warning_class, \*args, \*\*kwargs) | Fail unless the given callable throws the specified warning.
[assert_string_equal](https://numpy.org/devdocs/reference/generated/numpy.testing.assert_string_equal.html#numpy.testing.assert_string_equal)(actual, desired) | Test if two strings are equal.

## Decorators

method | description
---|---
[decorators.deprecated](https://numpy.org/devdocs/reference/generated/numpy.testing.decorators.deprecated.html#numpy.testing.decorators.deprecated)([conditional]) | Filter deprecation warnings while running the test suite.
[decorators.knownfailureif](https://numpy.org/devdocs/reference/generated/numpy.testing.decorators.knownfailureif.html#numpy.testing.decorators.knownfailureif)(fail_condition[, msg]) | Make function raise KnownFailureException exception if given condition is true.
[decorators.setastest](https://numpy.org/devdocs/reference/generated/numpy.testing.decorators.setastest.html#numpy.testing.decorators.setastest)([tf]) | Signals to nose that this function is or is not a test.
[decorators.skipif](https://numpy.org/devdocs/reference/generated/numpy.testing.decorators.skipif.html#numpy.testing.decorators.skipif)(skip_condition[, msg]) | Make function raise SkipTest exception if a given condition is true.
[decorators.slow](https://numpy.org/devdocs/reference/generated/numpy.testing.decorators.slow.html#numpy.testing.decorators.slow)(t) | Label a test as ‘slow’.
[decorate_methods](https://numpy.org/devdocs/reference/generated/numpy.testing.decorate_methods.html#numpy.testing.decorate_methods)(cls, decorator[, testmatch]) | Apply a decorator to all methods in a class matching a regular expression.

## Test Running

method | description
---|---
[Tester](https://numpy.org/devdocs/reference/generated/numpy.testing.Tester.html#numpy.testing.Tester) | alias of numpy.testing._private.nosetester.NoseTester
[run_module_suite](https://numpy.org/devdocs/reference/generated/numpy.testing.run_module_suite.html#numpy.testing.run_module_suite)([file_to_run, argv]) | Run a test module.
[rundocs](https://numpy.org/devdocs/reference/generated/numpy.testing.rundocs.html#numpy.testing.rundocs)([filename, raise_on_error]) | Run doctests found in the given file.
[suppress_warnings](https://numpy.org/devdocs/reference/generated/numpy.testing.suppress_warnings.html#numpy.testing.suppress_warnings)([forwarding_rule]) | Context manager and decorator doing much the same as warnings.catch_warnings.

## Guidelines

- [Testing Guidelines](https://www.numpy.org/devdocs/reference/testing.html)
  - [Introduction](https://www.numpy.org/devdocs/reference/testing.html#introduction)
  - [Writing your own tests](https://www.numpy.org/devdocs/reference/testing.html#writing-your-own-tests)
    - [Labeling tests](https://www.numpy.org/devdocs/reference/testing.html#labeling-tests)
    - [Easier setup and teardown functions / methods](https://www.numpy.org/devdocs/reference/testing.html#easier-setup-and-teardown-functions-methods)
    - [Parametric tests](https://www.numpy.org/devdocs/reference/testing.html#parametric-tests)
    - [Doctests](https://www.numpy.org/devdocs/reference/testing.html#doctests)
    - [tests/](https://www.numpy.org/devdocs/reference/testing.html#tests)
    - [\__init__.py and setup.py](https://www.numpy.org/devdocs/reference/testing.html#init-py-and-setup-py)
  - [Tips & Tricks](https://www.numpy.org/devdocs/reference/testing.html#tips-tricks)
    - [Creating many similar tests](https://www.numpy.org/devdocs/reference/testing.html#creating-many-similar-tests)
    - [Known failures & skipping tests](https://www.numpy.org/devdocs/reference/testing.html#known-failures-skipping-tests)
    - [Tests on random data](https://www.numpy.org/devdocs/reference/testing.html#tests-on-random-data)
