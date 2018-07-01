# 错误处理

Universal functions can trip special floating-point status registers in your hardware (such as divide-by-zero). If available on your platform, these registers will be regularly checked during calculation. Error handling is controlled on a per-thread basis, and can be configured using the functions

- ``seterr``([all, divide, over, under, invalid])	Set how floating-point errors are handled.
- ``seterrcall``(func)	Set the floating-point error callback function or log object.
