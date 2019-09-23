---
meta:
  - name: keywords
    content: NumPy 中的微分神经计算
  - name: description
    content: 可微分神经计算的实现尽可能接近于本文的描述。任务：char-level 预测。报告还包括简单RNN(RNN-numpy.py)和LSTM(LSTM-numpy.py)。
---

# NumPy 中的微分神经计算

可微分神经计算 https://www.nature.com/article/nature20101 的实现尽可能接近于本文的描述。任务：char-level 预测。报告还包括简单RNN(RNN-numpy.py)和LSTM(LSTM-numpy.py)。一些外部数据(ptb、wiki)需要单独下载。

## 译者前言

本文的项目的作者是一个老外 [krocki](https://github.com/krocki)，关于dnc、rnn、lstm 的实现源码都在他的github仓库 [https://github.com/krocki/dnc](https://github.com/krocki/dnc)。

## 快速开始

```python
python dnc-debug.py
```

这些版本已完成。

```python
python rnn-numpy.py
python lstm-numpy.py
python dnc-numpy.py
```

### 积分

RNN代码基于A.Karpath(min-char-rnn.py)的原始工作

gist: https://gist.github.com/karpathy/d4dee566867f8291f086

文章: http://karpathy.github.io/2015/05/21/rnn-effectiveness/

### 特性

- RNN版本仍然依赖numpy
- 添加批处理
- 将RNN修改为LSTM
- 包括梯度检测

### DNC 

**实施**

- LSTM控制器
- 2D存储器数组
- 内容可寻址的读/写

**问题**

关键相似度的softmax会导致崩溃（除以0） - 如果遇到这种情况，需要重新启动

**将要做**

- 动态内存分配/自由
- 更快的实现（使用PyTorch？）
- 保存模型
- 例子

### 示例输出：

时间，迭代，BPC（预测误差 - >每个字符的位数，越低越好），处理速度

```python
0: 4163.009 s, iter 104800, 1.2808 BPC, 1488.38 char/s
```

### 模型中的样本（alice29.txt）：


```python
 e garden as she very dunced.
                  
  Alice fighting be it.  The breats?
              here on likegs voice withoup.
                                                                               
  `You minced more hal disheze, and I done hippertyou-sage, who say it's a look down whales that
his meckling moruste!' said Alice's can younderen, in they puzzled to them!'
     
  `Of betinkling reple bade to, punthery pormoved the piose himble, of to he see foudhed
just rounds, seef wance side pigs, it addeal sumprked.
                                                                                    
  `As or the Gryphon,' Alice said,
Fith didn't begun, and she garden as in a who tew.'
    
  Hat hed think after as marman as much the pirly
startares to dreaps
was one poon it                                                                           
out him were brived they                                                        
proce?                                                                                    
                                                                                 
                                                                                          
  CHAT, I fary,' said the Hat,' said the Divery tionly to himpos.'               
                                                                                          
  `Com, planere?"'                                                               
                                                                                          
  `Ica--'                                                                        
            Onlice IN's tread!  Wonderieving again, `but her rist,' said Alice.           
                                                                                 
                                                                                          
  She                                                                            
sea do voice.                                                                             
                                                                                 
  `I'mm the Panthing alece of the when beaning must anquerrouted not reclow, sobs to      
                                                                                 
  `In of queer behind her houn't seemed                                                   
```

### 检查反向传递的数值梯度（最右边的列应该具有值<1e-4）;

中间列具有计算的分析和数值梯度范围（这些应该更多/更少）

```python
----
GRAD CHECK

Wxh:            n = [-1.828500e-02, 5.292866e-03]       min 3.005175e-09, max 3.505012e-07
                a = [-1.828500e-02, 5.292865e-03]       mean 5.158434e-08 # 10/4
Whh:            n = [-3.614049e-01, 6.580141e-01]       min 1.549311e-10, max 4.349188e-08
                a = [-3.614049e-01, 6.580141e-01]       mean 9.340821e-09 # 10/10
Why:            n = [-9.868277e-02, 7.518284e-02]       min 2.378911e-09, max 1.901067e-05
                a = [-9.868276e-02, 7.518284e-02]       mean 1.978080e-06 # 10/10
Whr:            n = [-3.652128e-02, 1.372321e-01]       min 5.520914e-09, max 6.750276e-07
                a = [-3.652128e-02, 1.372321e-01]       mean 1.299713e-07 # 10/10
Whv:            n = [-1.065475e+00, 4.634808e-01]       min 6.701966e-11, max 1.462031e-08
                a = [-1.065475e+00, 4.634808e-01]       mean 4.161271e-09 # 10/10
Whw:            n = [-1.677826e-01, 1.803906e-01]       min 5.559963e-10, max 1.096433e-07
                a = [-1.677826e-01, 1.803906e-01]       mean 2.434751e-08 # 10/10
Whe:            n = [-2.791997e-02, 1.487244e-02]       min 3.806438e-08, max 8.633199e-06
                a = [-2.791997e-02, 1.487244e-02]       mean 1.085696e-06 # 10/10
Wrh:            n = [-7.319636e-02, 9.466716e-02]       min 4.183225e-09, max 1.369062e-07
                a = [-7.319636e-02, 9.466716e-02]       mean 3.677372e-08 # 10/10
Wry:            n = [-1.191088e-01, 5.271329e-01]       min 1.168224e-09, max 1.568242e-04
                a = [-1.191088e-01, 5.271329e-01]       mean 2.827306e-05 # 10/10
bh:             n = [-1.363950e+00, 9.144058e-01]       min 2.473756e-10, max 5.217119e-08
                a = [-1.363950e+00, 9.144058e-01]       mean 7.066159e-09 # 10/10
by:             n = [-5.594528e-02, 5.814085e-01]       min 1.604237e-09, max 1.017124e-05
                a = [-5.594528e-02, 5.814085e-01]       mean 1.026833e-06 # 10/10
```


## 文章出处

由NumPy中文文档翻译，原作者为 krocki，翻译至：[https://github.com/krocki/dnc](https://github.com/krocki/dnc)

关于dnc、rnn、lstm 的实现源码都在[https://github.com/krocki/dnc](https://github.com/krocki/dnc)。