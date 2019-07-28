# 确定输出类型

如果所有输入参数都不是``ndarrays``，则ufunc（及其方法）的输出不一定是``ndarray``。实际上，如果任何输入定义了一个``__array_ufunc__``方法，那么控制将完全传递给该函数，即ufunc被覆盖。

如果没有任何输入覆盖ufunc，那么所有输出数组将被传递给输入的``__array_prepare__``和``__array_wrap__``方法（除了ndarrays和scalars），它定义了它并具有最高的``__array_priority__ ``通用功能的任何其他输入。 ndarray的默认``__array_priority__``是0.0，子类型的默认``__array_priority__``是1.0。矩阵的``__array_priority__``等于10.0。

所有ufunc也可以获取输出参数。如有必要，输出将转换为提供的输出数组的数据类型。如果输出使用带有``__array__``方法的类，则结果将写入``__array__``返回的对象。然后，如果该类也有一个``__array_prepare__``方法，则调用它，以便可以根据ufunc的上下文确定元数据（由ufunc本身组成的上下文，传递给ufunc的参数和ufunc域） 。）由``__array_prepare__``返回的数组对象被传递给ufunc进行计算。最后，如果类也有一个``__array_wrap__``方法，返回的ndarray结果将在将控制权传递回调用者之前传递给该方法。