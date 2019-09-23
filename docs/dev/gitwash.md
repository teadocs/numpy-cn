---
meta:
  - name: keywords
    content: NumPy Git 教程
  - name: description
    content: 这个章节描述了一般的 git 和 github 工作流。
---

# Git 教程

这个章节描述了一般的 [git](https://git-scm.com/) 和 [github](https://github.com/numpy/numpy) 工作流。

这不是全面的 [git](https://git-scm.com/) 参考。
它是为 [github](https://github.com/numpy/numpy) 托管服务量身定制的。
您可能会找到更好或更快的方法来完成[git](https://git-scm.com/)的工作，但这些应该可以让您开始。

有关学习 [git](https://git-scm.com/) 的一般资源，请参阅[其他git资源](https://numpy.org/devdocs/dev/gitwash/git_resources.html#git-resources)。

查看[GitHub帮助](https://help.github.com/)中提供的[github](https://github.com/numpy/numpy)安装帮助页面

## 安装 git

使用git进行开发完全不需要github。
Git是一个分布式版本控制系统。
为了在您的机器上使用git，您必须[安装它](https://git-scm.com/downloads)。

## 获取代码的本地副本

在命令行中：

``` bash
git clone git://github.com/numpy/numpy.git
```

现在，您在新的numpy目录中有了代码树的副本。
如果这样不起作用，您可以尝试使用其他只读网址：

``` bash
git clone https://github.com/numpy/numpy.git
```

## 更新代码

您可能不时需要提取最新的代码。 为此，请执行以下操作：

``` bash
cd numpy
git fetch
git merge --ff-only
```

现在，``numpy`` 中的分支树将具有初始存储库中的最新更改记录。

## Git开发入门

本节和下一节将详细描述如何设置git以使用NumPy源代码。
如果您已经设置了git，
请跳至[开发工作流程](development_workflow.html)。

### 基本的Git设置

- [安装 git](https://matplotlib.org/devel/gitwash/git_install.html#install-git).
- 向Git介绍你自己：

``` bash
git config --global user.email you@yourdomain.example.com
git config --global user.name "Your Name Comes Here"
```

### 创建属于你自己的 NumPy 代码

您只需要这样做一次。
这里的说明与 [http://help.github.com/forking/](http://help.github.com/forking/) 上的说明非常相似 - 请参阅该页面以获取更多详细信息。
我们在这里重复其中的一些只是为了提供 [NumPy](https://www.numpy.org/) 项目的细节，并建议一些默认名称。

#### 设置和配置github帐户

如果您没有 [github](https://github.com/numpy/numpy) 帐户，请转到 [github](https://github.com/numpy/numpy) 页面并创建一个。

然后，您需要配置您的帐户以允许写访问-请参阅[github帮助](https://help.github.com/)上的 ``生成SSH密钥`` 帮助。

#### 创建自己的 NumPy 分叉副本

1. 登录到您的 [github](https://github.com/numpy/numpy) 帐户。
1. 转到位于[NumPy](https://www.numpy.org/) github上的 [NumPy github](https://github.com/numpy/numpy) 主页。
1. 点击页面上名为 *fork* 的按钮:

![fork](/static/images/forking_button.png)

稍等片刻后，您应该就能在你自己的主页上找到你自己的 [NumPy](https://www.numpy.org/) 分叉副本。

### 设置你的Fork

首先，您要遵循有关[创建属于你自己的 NumPy 代码](#创建属于你自己的-numpy-代码)说明。

#### 概览

``` bash
git clone https://github.com/your-user-name/numpy.git
cd numpy
git remote add upstream https://github.com/numpy/numpy.git
```

#### 详细

##### 克隆你的fork仓库

1. 使用 ``git clone https://github.com/your-user-name/numpy.git`` 将分叉克隆到你的本地计算机。
1. 查看一下，然后将目录更改为新的repo：``cd numpy``。然后 ``git branch -a``向您显示所有分支。您将得到如下内容：

    ``` bash
    * master
    remotes/origin/master
    ```

    这告诉您您当前在 ``master`` 分支上，并且还具有到 ``origin/master`` 的远程连接。
    哪个远程存储库是 ``origin/master`` ？尝试使用 ``git remote -v`` 查看远程URL。
    他们将指向您的 [github](https://github.com/numpy/numpy) 分支。

    现在，您想连接到上游的 [NumPy github](https://github.com/numpy/numpy) 存储库，以便可以合并主干中的更改。

##### 将您的存储库链接到上游仓库

``` bash
cd numpy
git remote add upstream https://github.com/numpy/numpy.git
```

``上游（upstream）`` 只是我们用来引用 [NumPy github](https://github.com/numpy/numpy) 上主要 [NumPy](https://www.numpy.org/) 存储库的任意名称。

只是为了您自己的满意，请使用 ``git remote -v show`` 向您展示一个新的 'remote'，为您提供以下信息：

``` bash
upstream     https://github.com/numpy/numpy.git (fetch)
upstream     https://github.com/numpy/numpy.git (push)
origin       https://github.com/your-user-name/numpy.git (fetch)
origin       https://github.com/your-user-name/numpy.git (push)
```

为了与NumPy中的更改保持同步，您需要设置存储库，以使其默认情况下从“上游”提取。 这可以通过以下方式完成：

``` bash
git config branch.master.remote upstream
git config branch.master.merge refs/heads/master
```

您可能还想轻松访问发送到NumPy存储库的所有拉取请求：

``` bash
git config --add remote.upstream.fetch '+refs/pull/*/head:refs/remotes/upstream/pr/*'
Your config file should now look something like (from ``$ cat .git/config``):
```

```
[core]
        repositoryformatversion = 0
        filemode = true
        bare = false
        logallrefupdates = true
        ignorecase = true
        precomposeunicode = false
[remote "origin"]
        url = https://github.com/your-user-name/numpy.git
        fetch = +refs/heads/*:refs/remotes/origin/*
[remote "upstream"]
        url = https://github.com/numpy/numpy.git
        fetch = +refs/heads/*:refs/remotes/upstream/*
        fetch = +refs/pull/*/head:refs/remotes/upstream/pr/*
[branch "master"]
        remote = upstream
        merge = refs/heads/master
```

## Git 配置

### 概览

您的个人 [git](https://git-scm.com/) 配置将保存在主目录的 ``.gitconfig`` 文件中。 这是一个示例 ``.gitconfig`` 文件：

```
[user]
        name = Your Name
        email = you@yourdomain.example.com

[alias]
        ci = commit -a
        co = checkout
        st = status -a
        stat = status -a
        br = branch
        wdiff = diff --color-words

[core]
        editor = vim

[merge]
        summary = true
```

您可以直接编辑此文件，也可以使用 ``git config --global`` 命令：

``` bash
git config --global user.name "Your Name"
git config --global user.email you@yourdomain.example.com
git config --global alias.ci "commit -a"
git config --global alias.co checkout
git config --global alias.st "status -a"
git config --global alias.stat "status -a"
git config --global alias.br branch
git config --global alias.wdiff "diff --color-words"
git config --global core.editor vim
git config --global merge.summary true
```

要在另一台计算机上进行设置，您可以复制 ``~/.gitconfig`` 文件，或运行上面的命令。

### 细节

#### 配置 user.name 和 user.email

最好告诉 [git](https://git-scm.com/) 您是谁，以标记您对代码所做的任何更改。 最简单的方法是从命令行：

``` bash
git config --global user.name "Your Name"
git config --global user.email you@yourdomain.example.com
```

这会将设置写入您的git配置文件，该文件现在应包含带有您的姓名和电子邮件的用户部分：

```
[user]
      name = Your Name
      email = you@yourdomain.example.com
```

当然，您需要用您的实际姓名和电子邮件地址替换 ``Your Name`` 和 ``you@yourdomain.example.com``。

#### 别名

您可能会受益于一些常用命令的别名。

例如，您可能希望能够将 ``git checkout`` 缩短为 ``git co``。
或者您可能想将 ``git diff --color-words`` 别名（给出diff格式正确的输出）的别名为 ``git wdiff``

以下 ``git config --global`` 命令：

``` bash
git config --global alias.ci "commit -a"
git config --global alias.co checkout
git config --global alias.st "status -a"
git config --global alias.stat "status -a"
git config --global alias.br branch
git config --global alias.wdiff "diff --color-words"
```

将在您的.gitconfig文件中创建一个别名部分，其内容如下：

```
[alias]
        ci = commit -a
        co = checkout
        st = status -a
        stat = status -a
        br = branch
        wdiff = diff --color-words
```

#### 编辑器

您可能还需要确保使用了您选择的编辑器

``` bash
git config --global core.editor vim
```

#### 合并

在合并时要强制执行摘要（再次需要 ``~/.gitconfig`` 文件）：

``` bash
[merge]
   log = true
```

或从命令行：

``` bash
git config --global merge.log true
```

## 差异规格中的两个和三个点

感谢Yarik Halchenko的解释。

想象一下一系列的提交A，B，C，D ...想象有两个分支，主题和母版。
当母版处于提交 “E” 状态时，您从母版中分离了主题。
提交的图形如下所示：

```
     A---B---C topic
     /
D---E---F---G master
```

然后：

``` bash
git diff master..topic
```

将输出从G到C的差（即受F和G的影响），而：

``` bash
git diff master...topic
```

只会在主题分支中输出差异（即仅A、B和C）。

## 其他Git资源

### 教程和摘要

- [github 帮助](https://help.github.com) 提供了一系列出色的操作指南。
- [learn.github](https://learn.github.com/) 提供了一系列出色的教程。
- [git 深入书籍](https://git-scm.com/book/) 是一本很好的有关git的深入书籍。
- [git 备忘录](http://cheat.errtheblog.com/s/git) 是一个页面，其中提供了常用命令的摘要。
- [git 用户手册](https://www.kernel.org/pub/software/scm/git/docs/user-manual.html)
- [git 教程](https://www.kernel.org/pub/software/scm/git/docs/gittutorial.html)
- [git 社区书籍](https://book.git-scm.com/)
- [git ready](http://www.gitready.com/) - 一系列不错的教程
- [git casts](http://www.gitcasts.com/) - 提供git操作方法的视频片段。
- [git magic](http://www-cs-students.stanford.edu/~blynn/gitmagic/index.html) - 具有中间细节的扩展介绍
- [git parable](http://tom.preston-werner.com/2009/05/19/the-git-parable.html) 易于阅读，解释了git背后的概念。
- 我们自己的 [git foundation](https://matthew-brett.github.com/pydagogue/foundation.html) 借鉴了 [git parable](http://tom.preston-werner.com/2009/05/19/the-git-parable.html).
- Fernando Perez 的 git 页面 - [Fernando’s git page](http://www.fperez.org/py4science/git.html) - 包含了许多链接和提示。
- 一个很好的但技术性的页面 [git concepts](http://www.eecs.harvard.edu/~cduan/technical/git/)
- [git svn 速成课程](https://git-scm.com/course/svn.html): [git](https://git-scm.com/) 还有 [subversion](http://subversion.tigris.org/)

### 进阶git工作流程

有很多使用 [git](https://git-scm.com/) 的方法。这是其他项目提出的一些经验法则：

- Linus Torvalds 的 [git 管理](http://kerneltrap.org/Linux/Git_Management)
- Linus Torvalds 的 [linux git 工作流](https://www.mail-archive.com/dri-devel@lists.sourceforge.net/msg39091.html)。摘要：使用git工具使您的编辑历史尽可能清晰；在您进行主动开发的分支中，尽可能少地从上游编辑中合并。

### 在线手册页

您可以使用（例如 ``git help push`` 或（类似的东西）``git push --help``）在自己的机器上获取这些信息，
但是为了方便起见，以下是一些常用命令的在线手册页：

- [git add](https://www.kernel.org/pub/software/scm/git/docs/git-add.html)
- [git branch](https://www.kernel.org/pub/software/scm/git/docs/git-branch.html)
- [git checkout](https://www.kernel.org/pub/software/scm/git/docs/git-checkout.html)
- [git clone](https://www.kernel.org/pub/software/scm/git/docs/git-clone.html)
- [git commit](https://www.kernel.org/pub/software/scm/git/docs/git-commit.html)
- [git config](https://www.kernel.org/pub/software/scm/git/docs/git-config.html)
- [git diff](https://www.kernel.org/pub/software/scm/git/docs/git-diff.html)
- [git log](https://www.kernel.org/pub/software/scm/git/docs/git-log.html)
- [git pull](https://www.kernel.org/pub/software/scm/git/docs/git-pull.html)
- [git push](https://www.kernel.org/pub/software/scm/git/docs/git-push.html)
- [git remote](https://www.kernel.org/pub/software/scm/git/docs/git-remote.html)
- [git status](https://www.kernel.org/pub/software/scm/git/docs/git-status.html)