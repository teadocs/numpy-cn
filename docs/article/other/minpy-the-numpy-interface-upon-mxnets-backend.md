# MinPy：MXNet后端的NumPy接口

Machine learning is now enjoying its golden age. In the past few years, its effectiveness has been proved by solving many traditionally hard problems in computer vision and natural language processing. At the same time, different machine learning frameworks came out to justify different needs. These frameworks, fall generally into two different categories: symbolic programming and imperative programming.

## Symbolic V.S. Imperative Programming

Symbolic and imperative programing are two different programming models. Imperative programming are represented by TensorFlow, MXNet’s symbol system, and Theano etc. In symbolic programming model, the execution of a neural network is comprised of two steps. The graph of the computational model needs to be defined first, then the defined graph is sent to execution. For example:

```python
import numpy as np
A = Variable('A')
B = Variable('B')
C = B * A
D = C + Constant(1)
# compiles the function
f = compile(D)
d = f(A=np.ones(10), B=np.ones(10)*2)
```

(Example taken from [[1]](http://mxnet.io/architecture/program_model.html))

The core idea here is that the computational graph is defined beforehand, which means the definition and the execution are separated. The advantage of symbolic programming is that it has specified precise computational boundary. Therefore, it is easier to adopt deep optimizations. However, symbolic programming has its limitation. First, it cannot gracefully work with control dependency. Second, it is hard to master a new symbolic language for a newcomer. Third, since the description and the execution of the computational graph are separated, it is difficult to relate execution to value instantiation in the computation.

So what about imperative programming? In fact most programmers have already known imperative programming quite well. The everyday C, Pascal, or Python code is almost all imperative. The fundamental idea is that every command is executed step by step, without a separated stage to define the computational graph. For example

```python
import numpy as np
a = np.ones(10)
b = np.ones(10) * 2
c = b * a
d = c + 1
```

(Example taken from [[1]](http://mxnet.io/architecture/program_model.html))

When the code executes ``c = b * a`` and ``d = c + 1``, they just run the actual computation. Compared to symbolic programming, imperative programming is much more flexible, since there is no separation of definition and execution. This is important for debugging and visualization. However, its computational boundary is not predefined, leading to harder system optimization. NumPy and Torch adapts imperative programming model.

MXNet is a “mixed” framework that aims to provide both symbolic and imperative style, and leaves the choice to its users. While MXNet has a superb symbolic programming subsystem, its imperative subsystem is not powerful enough compared to other imperative frameworks such as NumPy and Torch. This leads to our goal: a fully functional imperative framework that focuses on flexibility without much performance loss, and works well with MXNet’s existing symbol system.

## What is MinPy

MinPy aims at providing a high performing and flexible deep learning platform, by prototyping a pure NumPy interface above MXNet backend. In one word, you get the following automatically with your NumPy code:

```python
import minpy.numpy as np
```

- Operators with GPU support will be ran on GPU.
- Graceful fallback for missing operations to NumPy on CPU.
- Automatic gradient generation with Autograd support.
- Seamless MXNet symbol integration.

## Pure NumPy, purely imperative

Why obsessed with NumPy interface? First of all, NumPy is an extension to the Python programming language, with support for large, multi-dimensional arrays, matrices, and a large library of high-level mathematical functions to operate on these abstractions. If you just begin to learn deep learning, you should absolutely start from NumPy to gain a firm grasp of its concepts (see, for example, the Stanford’s [CS231n course](http://cs231n.stanford.edu/syllabus.html)). For quick prototyping of advanced deep learning algorithms, you may often start composing with NumPy as well.

Second, as an extension of Python, your implementation follows the intuitive imperative style. This is the **only** style, and there is **no** new syntax constructs to learn. To have a taste of this, let’s look at some examples below.

## Printing and Debugging

![调试和打印变量](/static/images/article/other-minipy-p1.png)

In symbolic programming, the control dependency before the print statement is required, otherwise the print operator will not appear on the critical dependency path and thus not being executed. In contrast, MinPy is simply NumPy, as straightforward as Python’s hello world.

## Data-dependent branches

![数据相关分支](/static/images/article/other-minipy-p2.png)

In symbolic programming, the ``lambda`` is required in each branch to lazily expand the dataflow graph during runtime, which can be quite confusing. Again, MinPy is NumPy, and you freely use the if statement anyway you like.

Tensorflow is just one typical example, many other packages (e.g. Theano, or even MXNet) have similar problems. The underlying reason is the trade-off between symbolic programming and imperative programming. Code in symbolic programs (Tensorflow and Theano) generates dataflow graph instead of performing concrete computation. This enables extensive optimizations, but requires reinventing almost all language constructs (like if and loop). Imperative programs (NumPy) generates dataflow graph along with the computation, enabling you freely query or use the value just computed. In MinPy, we use NumPy syntax to ease your programming, while simultaneously achieving good performance.

## Dynamic automatic gradient computation

Automatic gradient computation has become essential in modern deep learning systems. In MinPy, we adopt [Autograd](https://github.com/HIPS/autograd)’s approach to compute gradients. Since the dataflow graph is generated along with the computation, all kinds of native control flow are supported during gradient computation. For example:

```python
import minpy
from minpy.core import grad

def foo(x):
  if x >= 0:
    return x
  else:
    return 2 * x

foo_grad = grad(foo)
print foo_grad(3)  # should print 1.0
print foo_grad(-1) # should print 2.0
```

Here, feel free to use native if statement. A complete tutorial about auto-gradient computation can be found [here](https://minpy.readthedocs.io/en/latest/tutorial/autograd_tutorial.html).

## Elegant fallback for missing operators

You never like ``NotImplementedError``, so do we. NumPy is a very large library. In MinPy, we automatically fallback to NumPy if some operators have not been implemented in MXNet yet. For example, the following code runs smoothly and you don’t need to worry about copying arrays back and forth from GPU to CPU; MinPy handles the fallback and its side effect transparently.

```python
import minpy.numpy as np
x = np.zeros((2, 3))     # Use MXNet GPU implementation
y = np.ones((2, 3))      # Use MXNet GPU implementation
z = np.logaddexp(x, y)   # Use NumPy CPU implementation
```

## Seamless MXNet symbol support

Although we pick the imperative side, we understand that symbolic programming is necessary for operators like convolution. Therefore, MinPy allows you to “wrap” a symbol into a function that could be called together with other imperative calls. From a programmer’s eye, these functions is just as other NumPy calls, thus we preserve the imperative style throughout:

```python
import mxnet as mx
import minpy.numpy as np
from minpy.core import Function
# Create Function from symbol.
net = mx.sym.Variable('x')
net = mx.sym.Convolution(net, name='conv', kernel=(3, 3), num_filter=32, no_bias=True)
conv = Function(net, input_shapes={'x', (8, 3, 10, 10)}
# Call Function as normal function.
x = np.zeros((8, 3, 10, 10))
w = np.ones((32, 3, 3, 3,))
y = np.exp(conv(x=x, conv_weight=w))
```

## Is MinPy fast?

The imperative interface does raise many challenges, especially it foregoes some of the deep optimization that only (currently) embodied in symbolic programming. However, MinPy manages to retain reasonable performance, especially when the actual computation is intense. Our next target is to get back the performance with advanced system techniques.

![基准，基准尺度](/static/images/article/benchmark.png)

## CS231n – a Perfect Intro for Newcomers

As for deep learning learners, MinPy is a perfect tool to begin with. One of the reasons is that MinPy is fully compatible with NumPy, which means almost no modification to the existing NumPy code. In addition, our team also provides a modified version of CS231n assignments to address the amenity of MinPy.

[CS231n](http://cs231n.stanford.edu/) is an introductory course to deep learning taught by Professor [Fei-Fei](http://vision.stanford.edu/feifeili/) Li and her Ph.D students [Andrej Karpathy](http://cs.stanford.edu/people/karpathy/) and [Justin Johnson](http://cs.stanford.edu/people/jcjohns/) at Stanford University. The curriculum covers all the basics of deep learning, including convolutional neural network and recurrent neural network. The course assignments are not only a simple practice of the lecture contents, but a in-depth journey for students to explore deep learning’s latest applications step-by-step. Since MinPy has similar interface with NumPy, our team modified assignments in CS231n to demonstrate the features MinPy provided, in order to create an integrated learning experience for the students who learn deep learning for their first time. The MinPy version of CS231n has already been used in deep learning course at Shanghai Jiao Tong University and ShanghaiTech University.

## Summary

MinPy development team began with [Minerva](https://github.com/dmlc/minerva) project. After co-founded MXNet project with the community, we have contributed to the core code of MXNet, including its executor engine, IO, and Caffe operator plugin. Having completed that part, the team decided to take a step back and rethink the user experience, before moving to yet another ambitious stage of performance optimizations. We strive to provide maximum flexibility for users, while creating space for more advanced system optimization. MinPy is pure NumPy and purely imperative. It will be merged into MXNet in the near future.

Enjoy! Please send us feedbacks.

## Links

- GitHub: https://github.com/dmlc/minpy
- MinPy documentation: http://minpy.readthedocs.io/en/latest/

## Acknowledgements

- [MXNet Community](http://dmlc.ml/)
- Sean Welleck (Ph.D at NYU), Alex Gai and Murphy Li (Undergrads at NYU Shanghai)
- Professor [Yi Ma](http://sist.shanghaitech.edu.cn/StaffDetail.asp?id=387) and Postdoc Xu Zhou at ShanghaiTech University; Professor [Kai Yu](https://speechlab.sjtu.edu.cn/~kyu/) and Professor [Weinan Zhang](http://wnzhang.net/) at Shanghai Jiao Tong University.

## MinPy Developers

- Project Lead Minjie Wang (NYU) [GitHub](https://github.com/jermainewang)
- Larry Tang (U of Michigan*) [GitHub](https://github.com/lryta)
- Yutian Li (Stanford) [GitHub](https://github.com/hotpxl)
- Haoran Wang (CMU) [GitHub](https://github.com/HrWangChengdu)
- Tianjun Xiao (Tesla) [GitHub](https://github.com/sneakerkg)
- Ziheng Jiang (Fudan U*) [GitHub](https://github.com/ZihengJiang)
- Professor [Zheng Zhang](https://shanghai.nyu.edu/academics/faculty/directory/zheng-zhang) (NYU Shanghai) [GitHub](https://github.com/zzhang-cn)

*: MinPy is completed during NYU Shanghai research internship

## Reference

http://mxnet.io/architecture/program_model.html

## 文章出处

由NumPy中文文档翻译，原作者为 Jan，翻译至：[http://dmlc.ml/2017/01/18/minpy-the-numpy-interface-upon-mxnets-backend.html](http://dmlc.ml/2017/01/18/minpy-the-numpy-interface-upon-mxnets-backend.html)