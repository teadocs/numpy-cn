# 浮点错误处理

## 设置和获取错误处理

- seterr([all, divide, over, under, invalid])	设置如何处理浮点错误。
- geterr()	获取当前处理浮点错误的方法。
- seterrcall(func)	设置浮点错误回调函数或日志对象。
- geterrcall()	返回用于浮点错误的当前回调函数。
- errstate(**kwargs)	用于浮点错误处理的上下文管理器。

## 内部功能

- seterrobj(errobj)	设置定义浮点错误处理的对象。
- geterrobj()	返回定义浮点错误处理的当前对象。