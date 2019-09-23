---
meta:
  - name: keywords
    content: NumPy 与 神经网络
  - name: description
    content: 对我来说用于分类的神经网络是一种非常可怕的机器学习算法。学习神经网络算法时，会给人一种望而生畏的感觉，但当我最终妥协并陷入其中无法自拔的时候...
---

# NumPy 与 神经网络

对我来说用于分类的神经网络是一种非常可怕的机器学习算法。学习神经网络算法时，会给人一种望而生畏的感觉，但当我最终妥协并陷入其中无法自拔的时候，才发现其实它并没有想象中的那么可怕。它们被称为神经网络，是因为它们松散地建立在人类大脑神经元以及神经元工作原理的基础上。但是，它们本质上是一组线性模型。关于这些算法的数学和结构有很多很好的文章来解释它们，所以这些部分我的这篇文章不会提及。相反，我将详细的用numpy库在python中编写一个一个的步骤，并非常清楚地解释它的。这篇文章的代码很大程度上基于[《集体智慧编程》](https://s.click.taobao.com/t?e=m%3D2%26s%3DXIetsYhTCu8cQipKwQzePOeEDrYVVa64K7Vc7tFgwiHjf2vlNIV67pZpQLiTO%2BhgmSMhGfkQJ77VdTmGfLKGc3msngnYL0uHYhNjQr6GXJQ0IVmWuK%2BMt0g0aHp6CeiC6hqtRuAxoUJbnlHS8Kikd9qH4uMbv1iQxgxdTc00KD8%3D&pvid=10_183.14.30.247_9333_1539405668948)中提供的神经网络代码，只要输入数据格式正确，我就稍微调整它以使其可用于任何数据集。

首先，我们可以将每个神经元视为具有激活功能。此功能确定神经元是 ``开`` 还是 ``关`` - 是否激活。我们将使用sigmoid函数，在逻辑回归中，它应该是非常见的函数。与逻辑回归不同，我们在使用神经网络时也需要sigmoid函数的导数。

```python
import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# derivative of sigmoid
# sigmoid(y) * (1.0 - sigmoid(y))
# the way we use this y is already sigmoided
def dsigmoid(y):
    return y * (1.0 - y)    
```

就像逻辑回归一样，神经网络中的Sigmoid函数将生成输入的端点(激活)乘以它们的权重。例如，假设我们有两列(特征)的输入数据和一个隐藏节点(神经元)在我们的神经网络。每个特征都会乘以相应的权重值，然后相加，然后通过S形(就像逻辑回归一样)。以这个简单的例子，并把它变成一个神经网络，我们只是添加更多的隐藏单元。除了添加更多的隐藏单元外，我们还将每个输入特性的路径添加到每个隐藏单元，并将其乘以相应的权重。每个隐藏单元取其输入*权值之和，并通过S形传递，从而导致该单元的激活。

接下来，我们将设置数组来保存用于网络的数据，并初始化一些参数。

```python
class MLP_NeuralNetwork(object):
    def __init__(self, input, hidden, output):
        """
        :param input: number of input neurons
        :param hidden: number of hidden neurons
        :param output: number of output neurons
        """
        self.input = input + 1 # add 1 for bias node
        self.hidden = hidden
        self.output = output
        # set up array of 1s for activations
        self.ai = [1.0] * self.input
        self.ah = [1.0] * self.hidden
        self.ao = [1.0] * self.output
        # create randomized weights
        self.wi = np.random.randn(self.input, self.hidden) 
        self.wo = np.random.randn(self.hidden, self.output) 
        # create arrays of 0 for changes
        self.ci = np.zeros((self.input, self.hidden))
        self.co = np.zeros((self.hidden, self.output))
```

我们要用矩阵做所有这些计算，因为它们速度快，而且非常容易阅读。我们的类将接受三个输入：输入层的大小(特性)、隐藏层的大小(要调优的变量参数)和输出层的数量(可能的类的数量)。我们设置一个1数组作为单元激活的占位符，一个0数组作为层更改的占位符。需要注意的一件重要事情是，我们将所有的权重初始化为随机数。重要的是权值是随机的，否则我们将无法调整网络。如果所有的权重是一样的，那么所有隐藏的单位都是一样的，那你的神经网络算法就废了。

所以现在是时候做一些预测的运算操作了。我们要做的是通过随机权重将所有数据通过网络提供给用户，并生成一些(不那么准确的)预测。后来，每次做出预测时，我们都会计算出预测的错误程度，以及为了使预测更好(即误差)，我们需要改变权重的方向。我们会做很多…很多次，当权重被更新时，我们会创建一个前馈函数，这个函数可以被一次又一次地调用。

```python
def feedForward(self, inputs):
   if len(inputs) != self.input-1:
        raise ValueError('Wrong number of inputs you silly goose!')
    # input activations
    for i in range(self.input -1): # -1 is to avoid the bias
        self.ai[i] = inputs[i]
    # hidden activations
    for j in range(self.hidden):
        sum = 0.0
        for i in range(self.input):
            sum += self.ai[i] * self.wi[i][j]
        self.ah[j] = sigmoid(sum)
    # output activations
    for k in range(self.output):
        sum = 0.0
        for j in range(self.hidden):
            sum += self.ah[j] * self.wo[j][k]
        self.ao[k] = sigmoid(sum)
    return self.ao[:]
```

输入激活只是输入功能。但是，对于另一层，激活变成了前一层激活的总和乘以它们的相应的权值，反馈到S形中去了。

在第一次运算之后，我们的预测的误差相当大的。所以我们将使用一个非常熟悉的概念，梯度下降。这是我感到兴奋的部分，因为我认为数学真的很聪明。与线性模型的梯度下降不同，我们需要对神经网络使用一点微积分。这就是为什么我们在开始的时候，为S函数的导数写了这个函数。

我们的反向传播算法首先计算我们预测的输出与真实输出的误差。然后我们在输出激活(预测值)上取S形的导数，以得到梯度的方向(斜率)，并将该值乘以误差。这就给了我们误差的大小，隐藏的权值需要改变哪个方向来修正它。然后我们进入到隐藏层，并根据前面计算的幅度和误差计算隐藏层权值的误差。

利用该误差和隐藏层激活的S形导数，我们计算了输入层的权重需要改变多少，以及在哪个方向上需要改变。

现在我们有了价值网络，我们想改变利率的多少，以及在什么方向上，我们真正做到了这一点。我们更新连接每一层的权重。我们通过将当前权重乘以学习速率常数以及相应的权重层的大小和方向来实现这一点。就像在线性模型中一样，我们使用学习速率常数在每一步中做一些小的改变，这样我们就有更好的机会为最小化成本函数的权值找到真正的值。

```python
def backPropagate(self, targets, N):
    """
    :param targets: y values
    :param N: learning rate
    :return: updated weights and current error
    """
    if len(targets) != self.output:
        raise ValueError('Wrong number of targets you silly goose!')
    # calculate error terms for output
    # the delta tell you which direction to change the weights
    output_deltas = [0.0] * self.output
    for k in range(self.output):
        error = -(targets[k] - self.ao[k])
        output_deltas[k] = dsigmoid(self.ao[k]) * error
    # calculate error terms for hidden
    # delta tells you which direction to change the weights
    hidden_deltas = [0.0] * self.hidden
    for j in range(self.hidden):
        error = 0.0
        for k in range(self.output):
            error += output_deltas[k] * self.wo[j][k]
        hidden_deltas[j] = dsigmoid(self.ah[j]) * error
    # update the weights connecting hidden to output
    for j in range(self.hidden):
        for k in range(self.output):
            change = output_deltas[k] * self.ah[j]
            self.wo[j][k] -= N * change + self.co[j][k]
            self.co[j][k] = change
    # update the weights connecting input to hidden
    for i in range(self.input):
        for j in range(self.hidden):
            change = hidden_deltas[j] * self.ai[i]
            self.wi[i][j] -= N * change + self.ci[i][j]
            self.ci[i][j] = change
    # calculate error
    error = 0.0
    for k in range(len(targets)):
        error += 0.5 * (targets[k] - self.ao[k]) ** 2
    return error
```

好的，让我们把它们链接在一起，创建训练和预测功能。训练网络的步骤是非常直接和直观的。我们首先调用“``前馈``”函数，它给出我们初始化的随机权值的输出。然后，我们调用反向传播算法来调整和更新权值，以做出更好的预测。然后再调用前馈函数，但这一次它使用了更新后的权值，预测结果略好一些。我们将这个循环保持在一个预先确定的迭代数量中，在此期间，我们应该看到错误下降到接近0。

```python
def train(self, patterns, iterations = 3000, N = 0.0002):
    # N: learning rate
    for i in range(iterations):
        error = 0.0
        for p in patterns:
            inputs = p[0]
            targets = p[1]
            self.feedForward(inputs)
            error = self.backPropagate(targets, N)
        if i % 500 == 0:
            print('error %-.5f' % error)
```

最后，对于预测操作。我们只是简单地调用前馈函数，它将返回输出层的激活。记住，每一层的激活是前一层输出的线性组合。

```python
def predict(self, X):
    """
    return list of predictions after training algorithm
    """
    predictions = []
    for p in X:
        predictions.append(self.feedForward(p))
    return predictions
```

基本上就是这样！你可以[在这里](https://github.com/FlorianMuellerklein/Machine-Learning/blob/master/Old/BackPropagationNN.py)看到完整的代码。

我运行了这个代码的数字识别数据集提供的skLearning，它完成了一个97%的准确性。我要说那是相当成功的！

```
            precision    recall  f1-score   support

          0       0.98      0.96      0.97        49
          1       0.92      0.97      0.95        36
          2       1.00      1.00      1.00        43
          3       0.95      0.88      0.91        41
          4       0.98      1.00      0.99        47
          5       0.96      1.00      0.98        46
          6       1.00      1.00      1.00        47
          7       0.98      0.96      0.97        46
          8       0.93      0.80      0.86        49
          9       1.00      0.91      0.95        46

avg / total       0.97      0.95      0.96       450
```

## 文章出处

由NumPy中文文档翻译，原作者为 Florian 和 Oliver 兄弟俩，翻译至：[https://databoys.github.io/Feedforward/](https://databoys.github.io/Feedforward/)