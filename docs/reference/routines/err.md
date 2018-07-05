# 浮点错误处理

## Setting and getting error handling

- seterr([all, divide, over, under, invalid])	Set how floating-point errors are handled.
- geterr()	Get the current way of handling floating-point errors.
- seterrcall(func)	Set the floating-point error callback function or log object.
- geterrcall()	Return the current callback function used on floating-point errors.
- errstate(**kwargs)	Context manager for floating-point error handling.

## Internal functions

- seterrobj(errobj)	Set the object that defines floating-point error handling.
- geterrobj()	Return the current object that defines floating-point error handling.