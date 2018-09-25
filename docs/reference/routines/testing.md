# 测试相关API

公共测试库为所有的numpy测试脚本提供测试支持。

这个单独的模块应该在一个位置提供numpy测试的所有常用功能，这样测试脚本就可以只导入它并立即工作。

## 断言

- assert_almost_equal(actual, desired[, …])	如果两个项目不等于所需精度，则引发AssertionError。
- assert_approx_equal(actual, desired[, …])	如果两个项目不等于有效数字，则引发AssertionError。
- assert_array_almost_equal(x, y[, decimal, …])	如果两个对象不等于所需的精度，则引发AssertionError。
- assert_allclose(actual, desired[, rtol, …])	如果两个对象不等于所需的容差，则引发AssertionError。
- assert_array_almost_equal_nulp(x, y[, nulp])	比较两个数组的间距。
- assert_array_max_ulp(a, b[, maxulp, dtype])	检查所有数组项是否在“最后的位置”中最多为N个单位。
- assert_array_equal(x, y[, err_msg, verbose])	如果两个数组类对象不相等，则引发AssertionError。
- assert_array_less(x, y[, err_msg, verbose])	如果两个类数组对象不是由小于排序的，则引发AssertionError。
- assert_equal(actual, desired[, err_msg, verbose])	如果两个对象不相等，则引发AssertionError。
- assert_raises(exception_class, callable, …)	失败，除非调用参数args和关键字参数kwargs时调用able_class抛出类的异常。
- assert_raises_regex(exception_class, …)	失败，除非在调用参数args和关键字参数kwargs时，调用able抛出与Expect_regexp匹配的异常类EXCEPTION_CLASE和WITH消息。
- assert_warns(warning_class, *args, **kwargs)	除非给定的可调用对象抛出指定的警告，否则将失败。
- assert_string_equal(actual, desired)	测试两个字符串是否相等。

## 装饰方法

- decorators.deprecated([conditional])	运行测试套件时过滤弃用警告。
- decorators.knownfailureif(fail_condition[, msg])	如果给定条件为真，则使函数引发KnownFailureException异常。
- decorators.setastest([tf])	发出嗅探信号返回此函数是否是测试函数。
- decorators.skipif(skip_condition[, msg])	如果给定条件为真，则使函数引发SkipTest异常。
- decorators.slow(t)	将测试标为'slow'。
- decorate_methods(cls, decorator[, testmatch])	将装饰器应用于与正则表达式匹配的类中的所有方法。

## 运行测试

- Tester	numpy.testing.nose_tools.nosetester.NoseTester别名
- run_module_suite([file_to_run, argv]) 运行测试模块。
- rundocs([filename, raise_on_error])	在指定的文件中找到并运行doctests。
- suppress_warnings([forwarding_rule])	上下文管理器和装饰器执行的操作与warnings.catch_warns大致相同。