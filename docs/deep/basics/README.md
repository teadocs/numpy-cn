---
meta:
  - name: keywords
    content: 深度学习
  - name: description
    content: 本章由9篇文档组成，它们按照简单到难的顺序排列，将指导您如何使用NumPy和PaddlePaddle完成基础的深度学习任务。
---

# 深度学习基础教程

本章由9篇文档组成，它们按照简单到难的顺序排列，将指导您如何使用 [NumPy](https://www.numpy.org.cn) 和 [PaddlePaddle](https://www.paddlepaddle.org.cn/?from=pandas-cn) 完成基础的深度学习任务。

本章文档涉及大量了深度学习基础知识，也介绍了如何使用 [NumPy](https://www.numpy.org.cn) 和 [PaddlePaddle](https://www.paddlepaddle.org.cn/?from=pandas-cn) 实现这些内容，请参阅以下说明了解如何使用：

## 内容简介

您现在在看的这本书是一本“交互式”电子书 —— 每一章都可以运行在一个Jupyter Notebook里。

- [线性回归](fit_a_line)
- [数字识别](fit_a_line)
- [图像分类](fit_a_line)
- [词向量](fit_a_line)
- [个性化推荐](fit_a_line)
- [情感分析](fit_a_line)
- [语义角色标注](fit_a_line)
- [机器翻译](fit_a_line)
- [生成对抗网络](fit_a_line)

我们把[NumPy](https://www.numpy.org.cn/)、Jupyter、PaddlePaddle、以及各种被依赖的软件都打包进一个Docker image了。
所以您不需要自己来安装各种软件，只需要安装Docker即可。
对于各种Linux发行版，请参考 [https://www.docker.com](https://www.docker.com/) 。如果您使用 [Windows](https://www.docker.com/docker-windows) 或者 [Mac](https://www.docker.com/docker-mac)，
可以考虑 [给Docker更多内存和CPU资源](http://stackoverflow.com/a/39720010/724872)。

## 使用方法

本书默认使用CPU训练，若是要使用GPU训练，使用步骤会稍有变化,请参考下文“使用GPU训练”

### 使用CPU训练

只需要在命令行窗口里运行：

``` bash
$ docker run -d -p 8888:8888 paddlepaddle/book
```

即可从DockerHub.com下载和运行本书的Docker image。阅读和在线编辑本书请在浏览器里访问 [http://localhost:8888](http://localhost:8888/)

如果您访问 DockerHub.com 很慢，可以试试我们的另一个镜像 docker.paddlepaddlehub.com：

``` bash
$ docker run -d -p 8888:8888 docker.paddlepaddlehub.com/book
```

### 使用GPU训练

为了保证GPU驱动能够在镜像里面正常运行，我们推荐使用 [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) 来运行镜像。请先安装nvidia-docker，之后请运行：

``` bash
$ nvidia-docker run -d -p 8888:8888 paddlepaddle/book:latest-gpu
```

或者使用国内的镜像请运行：

``` bash
$ nvidia-docker run -d -p 8888:8888 docker.paddlepaddlehub.com/book:latest-gpu
```

还需要将以下代码

``` python
use_cuda = False
```

改成：

``` python
use_cuda = True
```

## 贡献新章节

您要是能贡献新的章节那就太好了！请发 Pull Requests 把您写的章节加入到 pending 下面的一个子目录里。当这一章稳定下来，我们一起把您的目录挪到根目录。

为了写作、运行、调试，您需要安装Python 2.x和Go >1.5, 并可以用 [脚本程序](https://github.com/PaddlePaddle/book/blob/develop/.tools/convert-markdown-into-ipynb-and-test.sh) 来生成新的Docker image。

Please Note: We also provide [English Readme](https://github.com/PaddlePaddle/book/blob/develop/README.md) for PaddlePaddle book