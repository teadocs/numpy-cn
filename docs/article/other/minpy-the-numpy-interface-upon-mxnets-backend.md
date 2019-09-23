---
meta:
  - name: keywords
    content: MinPy：MXNet后端的NumPy接口
  - name: description
    content: 机器学习现在正处于黄金时期。在过去的几年中，它的有效性已经被计算机视觉和自然语言处理中的许多传统难题所证明。同时，出现了不同的机器学习框架来证明不同的需求。这些框架一般分为两类：``符号编程(Symbolic)`` 和 ``命令式编程(Imperative Programming)``。
---

# MinPy：MXNet后端的NumPy接口

机器学习现在正处于黄金时期。在过去的几年中，它的有效性已经被计算机视觉和自然语言处理中的许多传统难题所证明。同时，出现了不同的机器学习框架来证明不同的需求。这些框架一般分为两类：``符号编程(Symbolic)`` 和 ``命令式编程(Imperative Programming)``。

## 符号编程(Symbolic) 比较 命令式编程(Imperative Programming)

符号编程和命令式编程是两种不同的编程模型。命令式编程以TensorFlow、MXNet的符号系统、Theano等为代表。在符号编程模型中，神经网络的执行由两步组成。首先需要定义计算模型的图，然后将定义的图发送给执行。例如：

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

(代码示例取自 [[1]](http://mxnet.io/architecture/program_model.html))

这里的核心思想是预先定义计算图，这意味着定义和执行是分开的。符号编程的优点在于它具有精确的计算边界。因此，更容易采用深度优化。然而，符号编程也有其局限性。首先，它不能很好地处理控制依赖关系。其次，对于新来者来说，要掌握一种新的象征性语言是很困难的。第三，由于计算图的描述和执行是分离的，在计算中很难将执行与值实例化联系起来。

那么命令式编程又如何呢？事实上，大多数程序员已经很好地了解了命令式编程。日常的C、Pascal或Python代码几乎都是必需的。基本思想是，每个命令都是一步执行的，没有一个单独的阶段来定义计算图。例如：

```python
import numpy as np
a = np.ones(10)
b = np.ones(10) * 2
c = b * a
d = c + 1
```

(代码示例取自 [[1]](http://mxnet.io/architecture/program_model.html))

当代码执行 ``C = b * a`` 和 ``D = c + 1`` 时，它们只是运行实际的计算。与符号编程相比，命令式编程更加灵活，因为没有定义和执行的分离。这对于调试和可视化非常重要。然而，它的计算边界并不是预先定义的，这就导致了更难进行系统优化。NumPy和Torch采用命令式编程模型。

MXNet是一个“混合”框架，旨在提供符号风格和命令风格，并将选择权留给用户。虽然MXNet有一个极好的符号编程子系统，但与其他命令框架(如NumPy和Torch)相比，它的命令式子系统还不够强大。这就导致了我们的目标：一个功能完备的命令式框架，它注重灵活性而不损失多少性能，并且与MXNet的现有符号系统很好地配合。

## 什么是MinPy

MinPy的目标是提供一个高性能和灵活的深度学习平台，通过原型化MXNet后端上的一个纯NumPy接口。总之，你将通过NumPy代码自动获得以下内容：

```python
import minpy.numpy as np
```

- 具有GPU支持的操作人员将在GPU上运行。
- 对CPU上的NumPy缺少操作的优雅回退。
- 自动梯度生成支持自动梯度。
- 无缝MXNet符号集成。

## 纯NumPy，纯命令

为什么我们要痴迷于NumPy的界面？首先，NumPy是Python编程语言的扩展，支持大型多维数组、矩阵和高级数学函数的大型库，用于对这些抽象进行操作。如果你刚刚开始学习深度学习，你绝对应该从NumPy开始，以便对它的概念有一个牢固的了解(例如，参见斯坦福大学的[CS231n课程](http://cs231n.stanford.edu/becus.html)。对于高级深度学习算法的快速原型，你通常也可以使用NumPy来编写程序。

其次，作为Python的扩展，你的实现遵循直观的命令式变成风格。这是 **唯一 only** 风格，并且 *不需要* 学习新的语法结构。为了尝试这一点，让我们看看下面的一些例子。

## 打印和调试

![调试和打印变量](/static/images/article/other-minipy-p1.png)

在符号编程中，需要在print语句之前的控制依赖性，否则打印操作符将不会出现在关键依赖路径上，因此不会被执行。相比之下，MinPy简直就是NumPy原生，就像Python的hello world一样简单。

## 数据相关分支

![数据相关分支](/static/images/article/other-minipy-p2.png)

在符号编程中，每个分支中都需要 ``lambda`` 表达式，以便在运行时缓慢地展开数据流图，这可能非常令人费解。同样，MinPy是NumPy，你可以随心所欲地使用if语句。

TensorFlow只是一个典型的例子，许多其他包(例如Theano，甚至MXNet)都有类似的问题。根本原因在于符号编程和命令式编程之间的权衡。符号程序(TensorFlow和Theano)中的代码生成数据流图，而不是执行具体的计算。这可以进行广泛的优化，但需要重新创建几乎所有的语言构造(如if和循环)。命令式程序(NumPy)与计算一起生成数据流图，你你可以自由地查询或使用刚刚计算的值。在MinPy中，我们使用NumPy语法来简化编程，同时也能获得良好的性能。

## 动态自动梯度计算

自动梯度计算已经成为现代深度学习系统中必不可少的一部分。在MinPy中，我们采用[Autograd](https://github.com/hips/autograd) 的方法计算梯度。由于数据流图是随计算生成的，因此在梯度计算中支持各种底层的控制流。例如：

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

在这里，你可以自由地使用本地if语句代码。关于自动梯度计算的完整教程可以在[这里](https://minpy.readthedocs.io/en/end/tutual.html)找到。

## 无效操作的备用解决方案

你从来不喜欢 ``NotImplementedError`` 的错误异常，我们也一样。NumPy是一个非常大的库。在MinPy中，如果某些运算符尚未在MXNet中实现，我们将自动回退到NumPy。例如，下面的代码运行顺利，尼不需要担心将数组从GPU复制到CPU；MinPy透明地处理后备及其副作用。

```python
import minpy.numpy as np
x = np.zeros((2, 3))     # Use MXNet GPU implementation
y = np.ones((2, 3))      # Use MXNet GPU implementation
z = np.logaddexp(x, y)   # Use NumPy CPU implementation
```

## 无缝MXNet符号支持

虽然我们选择了命令式，但我们知道符号编程对于像卷积这样的操作符是必要的。因此，MinPy允许你将一个符号 “包装” 到一个可以与其他命令式调用一起调用的函数中。从程序员的角度来看，这些函数与其他NumPy调用一样，因此我们在整个过程中都保留了命令式风格：

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

## MinPy速度快吗？

命令式的接口确实有许多的挑战，特别是它放弃了一些(目前)仅体现在符号编程中的深度优化。然而，MinPy设法保持了合理的性能，特别是在实际计算量很大的情况下。我们的下一个目标是用先进的系统技术恢复性能。

![基准，基准尺度](/static/images/article/benchmark.png)

## CS231n – 一个针对新手的完美介绍

对于深度学习的学习者来说，MinPy是一个非常好的工具。其中一个原因是MinPy与NumPy完全兼容，这意味着几乎不需要修改现有的NumPy代码。此外，我们的团队还提供了CS231n作业的修改版本，以让大家更好的适应MinPy。

[CS231n](http://cs231n.stanford.edu/)是一门深度学习入门课程，由斯坦福大学的[李飞飞](http://vision.stanford.edu/feifeili/)教授和她的博士[Andrej Karpathy](http://cs.stanford.edu/people/karpathy/)和[Justin Johnson](http://cs.stanford.edu/people/jcjohns/)教授制作。课程涵盖了深度学习的所有基础知识，包括卷积神经网络和递归神经网络。这门课的作业不仅仅是对课程内容的简单练习，而是一次从浅入深的旅程，让学生一步一步地探索深度学习的最新应用。由于MinPy与NumPy具有相似的界面，我们的团队修改了CS231n中的作业，以演示MinPy提供的特性，以便为第一次学习深度学习的学生创建一个完整的学习体验。CS231n的MinPy版本已经在上海交通大学和上海理工大学的深造课程中得到了应用。

## 总结

MinPy开发团队从[Minerva](https://github.com/dmlc/minerva)项目开始。在与社区共同创建MXNet项目后，我们为MXNet的核心代码做出了贡献，包括它的执行器引擎、IO和CaffeOperator插件。在完成这部分之后，团队决定后退一步，重新考虑用户体验，然后再进入另一个雄心勃勃的性能优化阶段。我们努力为用户提供最大的灵活性，同时为更高级的系统优化创造空间。MinPy是纯粹的NumPy和纯粹的命令。它将在不久的将来并入MXNet。

好好享受吧！请给我们反馈。

## 链接资源

- GitHub: https://github.com/dmlc/minpy
- MinPy 英文文档: http://minpy.readthedocs.io/en/latest/

## 鸣谢

- [MXNet 社区](http://dmlc.ml/)
- Sean Welleck (纽约大学博士), Alex Gai and Murphy Li (纽约大学上海分校本科生)
- 上海理工大学教授[Yi Ma](http://sist.shanghaitech.edu.cn/StaffDetail.asp?id=387)、博士后Xu Zhou、上海交通大学教授[Kai Yu](https://speechlab.sjtu.edu.cn/~kyu/)、[Weinan Zhang](http://wnzhang.net/)教授。

## MinPy 开发者

- 项目负责人 Minjie Wang (NYU) [GitHub](https://github.com/jermainewang)
- Larry Tang (密歇根大学) [GitHub](https://github.com/lryta)
- Yutian Li (斯坦福大学) [GitHub](https://github.com/hotpxl)
- Haoran Wang (CMU) [GitHub](https://github.com/HrWangChengdu)
- Tianjun Xiao (Tesla) [GitHub](https://github.com/sneakerkg)
- Ziheng Jiang (复旦大学) [GitHub](https://github.com/ZihengJiang)
- 教授 [Zheng Zhang](https://shanghai.nyu.edu/academics/faculty/directory/zheng-zhang) (上海纽约大学) [GitHub](https://github.com/zzhang-cn)

*: MinPy在纽约大学上海研究实习期间完成

## 参考

http://mxnet.io/architecture/program_model.html

## 文章出处

由NumPy中文文档翻译，原作者为 Jan，翻译至：[http://dmlc.ml/2017/01/18/minpy-the-numpy-interface-upon-mxnets-backend.html](http://dmlc.ml/2017/01/18/minpy-the-numpy-interface-upon-mxnets-backend.html)