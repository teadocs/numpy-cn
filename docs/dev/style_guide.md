---
meta:
  - name: keywords
    content: NumPy C风格指南
  - name: description
    content: NumPy C编码约定基于Guido van Rossum的Python PEP-0007，并增加了一些限制。
---

# NumPy C风格指南

NumPy C编码约定基于Guido van Rossum的Python PEP-0007，并增加了一些限制。有许多C编码惯例，必须强调的是，NumPy惯例的主要目标不是选择“最佳”，但肯定存在分歧，而是要达到一致性。由于NumPy约定与PEP-0007中的约定非常接近，因此将PEP用作下面的模板，其中NumPy在适当的位置添加和变化。

## 介绍

本文档给出了包含NumPy的C实现的C代码的编码约定。注意，规则是​​有缺陷的。打破特定规则的两个充分理由：

1. 应用规则会使代码的可读性降低，即使对于习惯于阅读遵循规则的代码的人也是如此。
1. 为了与周围的代码保持一致，这些代码也会破坏它（可能是出于历史原因） - 尽管这也是一个清理别人混乱的机会。

## C方言

- 使用C99（即ISO / IEC 9899：1999定义的标准）。
- 不要使用GCC扩展（例如，不要在没有尾部反斜杠的情况下编写多行字符串）。最好将长串打破到单独的行上，如下所示：

``` python
"blah blah"
"blah blah"
```

这将适用于MSVC，否则会在非常长的字符串上窒息。
- 所有函数声明和定义必须使用完整原型（即指定所有参数的类型）。
- 没有编译器警告主要编译器（gcc，VC ++，其他一些）。注意：NumPy仍然会产生需要解决的编译器警告。

## 代码布局

- 使用4个空格的缩进，根本没有标签。
- 任何行都不应超过80个字符。如果这和前面的规则一起没有给你足够的代码空间，那么你的代码太复杂了，考虑使用子程序。
- 任何行都不应该以空格结尾。如果您认为需要重要的尾随空格，请再想一想，某人的编辑可能会将其删除为常规问题。
- 函数定义样式：第1列中的函数名，第1列中最外面的花括号，局部变量声明后的空行：

``` python
static int
extra_ivars(PyTypeObject *type, PyTypeObject *base)
{
    int t_size = PyType_BASICSIZE(type);
    int b_size = PyType_BASICSIZE(base);

    assert(t_size >= b_size); /* type smaller than base! */
    ...
    return 1;
}
```

如果转换到C ++，则可能会放宽此表单，以便内联的短类方法可以在与函数名称相同的行上具有返回类型。但是，这还有待确定。
- 代码结构：关键字之间有一个空格``if``，``for``左下括号; 括号内没有空格; 所有``if``分支周围都有括号，并且没有与该行相同的语句
 ``if``。它们的格式应如下所示：

``` python
if (mro != NULL) {
    one_line_statement;
}
else {
    ...
}


for (i = 0; i < n; i++) {
    one_line_statement;
}


while (isstuff) {
    dostuff;
}


do {
    stuff;
} while (isstuff);


switch (kind) {
    /* Boolean kind */
    case 'b':
        return 0;
    /* Unsigned int kind */
    case 'u':
        ...
    /* Anything else */
    default:
        return 3;
}
```
- return语句应该 *不会* 让多余的括号：

``` python
return Py_None; /* correct */
return(Py_None); /* incorrect */
```
- 函数和宏调用样式：打开paren之前没有空格，parens里面没有空格，逗号前没有空格，每个逗号后面有一个空格。``foo(a, b, c)``
- 始终在赋值，布尔和比较运算符周围放置空格。在使用大量运算符的表达式中，在最外层（最低优先级）运算符周围添加空格。
- 打破长行：如果可以的话，在最外面的参数列表中用逗号分隔。始终适当地缩进延续线，例如，

``` python
PyErr_SetString(PyExc_TypeError,
        "Oh dear, you messed up.");
```

这里适当地表示至少两个选项卡。没有必要用函数调用的左括号来排列所有内容。
- 当您在二元运算符处断开长表达式时，运算符将在前一行的末尾处运行，例如，

``` python
if (type > tp_dictoffset != 0 &&
        base > tp_dictoffset == 0 &&
        type > tp_dictoffset == b_size &&
        (size_t)t_size == b_size + sizeof(PyObject *)) {
    return 0;
}
```

请注意，多行布尔表达式中的术语是缩进的，以便使代码块的开头清晰可见。
- 在函数，结构定义和函数内的主要部分周围放置空行。
- 评论在他们描述的代码之前。多行注释应如下：

``` python
/*
 * This would be a long
 * explanatory comment.
 */
```

应谨慎使用尾随评论。代替

``` python
if (yes) { // Success!
```

做

``` python
if (yes) {
    // Success!
```
- 当在当前编译单元之外不需要时，应将所有函数和全局变量声明为静态。
- 在头文件中声明外部函数和变量。

## 命名约定

- 目前已为NumPy的公共职能没有统一的前缀，但他们都开始用某种类型的前缀，其次是下划线，并在驼峰：``PyArray_DescrAlignConverter``，``NpyIter_GetIterNext``。在未来，名称应该是形式``Npy*_PublicFunction``，明星是适当的。
- ``NPY_``例如，公共宏应该有一个前缀，然后使用大写``NPY_DOUBLE``。
- 私有函数应该是带有下划线的小写，例如：
 ``array_real_get``。不应使用单个前导下划线，但由于历史事故，某些当前功能名称违反了该规则。这些功能应该在某个时候重命名。

## 功能文档

NumPy目前没有C函数文档标准，但需要一个。代码中没有记录大多数numpy函数，而且应该更改。一种可能性是带有插件的Doxygen，因此用于Python函数的相同NumPy样式也可用于记录C函数，请参阅doc / cdoc /中的文件。