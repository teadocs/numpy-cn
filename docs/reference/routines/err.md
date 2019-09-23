# 浮点错误处理

## 设置和获取错误处理

方法 | 描述
---|---
[seterr](https://numpy.org/devdocs/reference/generated/numpy.seterr.html#numpy.seterr)([all, divide, over, under, invalid]) | 设置如何处理浮点错误。
[geterr](https://numpy.org/devdocs/reference/generated/numpy.geterr.html#numpy.geterr)() | 获取当前处理浮点错误的方法。
[seterrcall](https://numpy.org/devdocs/reference/generated/numpy.seterrcall.html#numpy.seterrcall)(func) | 设置浮点错误回调函数或日志对象。
[geterrcall](https://numpy.org/devdocs/reference/generated/numpy.geterrcall.html#numpy.geterrcall)() | 返回用于浮点错误的当前回调函数。
[errstate](https://numpy.org/devdocs/reference/generated/numpy.errstate.html#numpy.errstate)(\*\*kwargs) | 用于浮点错误处理的上下文管理器。

## 内部功能

方法 | 描述
---|---
[seterrobj](https://numpy.org/devdocs/reference/generated/numpy.seterrobj.html#numpy.seterrobj)(errobj) | 设置定义浮点错误处理的对象。
[geterrobj](https://numpy.org/devdocs/reference/generated/numpy.geterrobj.html#numpy.geterrobj)() | 返回定义浮点错误处理的当前对象。