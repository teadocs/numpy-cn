---
meta:
  - name: keywords
    content: 给 NumPy 做贡献
  - name: description
    content: 不是程序员？不是问题！NumPy是多方面的，我们有很多地方需要帮助。以下这些都是我们需要获得的帮助...
---

# 给 NumPy 做贡献

不是程序员？不是问题！NumPy是多方面的，我们有很多地方需要帮助。以下这些都是我们需要获得的帮助（它们都非常重要，因此我们按字母顺序列出）：

- 代码维护和开发
- 社区协调
- DevOps
- 开发教育内容和叙述文档
- 编写技术文档
- 筹款
- 项目管理
- 市场或宣传
- 翻译内容
- 网站设计与开发

本文档的其余部分讲述了NumPy代码库和文档的工作。我们正在更新我们对更多的贡献工作和角色分工的描述。如果您对这些其他贡献工作感兴趣，请联系我们！您可以通过[numpy-discussion 邮件列表](https://scipy.org/scipylib/mailing-lists.html)或 GitHub（打开问题或评论相关问题）来完成此操作。这些是我们首选的沟通渠道（开源本质上是公开的！），但如果您想首先私下讨论，请联系我们的社区协调员 *numpy-team@googlegroups.com* 或 *numpy-team.slack.com*（首次发送电子邮件至 *numpy-team@googlegroups.com* 以获取邀请。

## 开发流程 - 摘要

以下是简短摘要，完整的TOC链接如下：

1. 如果您是新手贡献者：

    - 前往 https://github.com/numpy/numpy 并单击 “fork” 按钮以创建您自己的项目副本。

    - 将项目克隆到本地计算机：

      ``` bash
      git clone https://github.com/your-username/numpy.git
      ```

    - 更改目录：

      ``` bash
      cd numpy
      ```

    - 添加远程仓库：

      ``` bash
      git remote add upstream https://github.com/numpy/numpy.git
      ```

    - 现在，``git remote -v`` 将显示两个名为的远程存储库：
        - upstream, 指的是 ``numpy`` 的仓库
        - origin, 指的是你的私人的fork而来的仓库
1. 进行贡献时

    - 从上游拉取最新的内容到你的本地：

    ``` bash
    git checkout master
    git pull upstream master
    ```

    - 为要处理的功能创建分支。由于分支名称将出现在合并消息中，因此请命名合理的名称，例如 “linspace-speedups”：

    ``` bash
    git checkout -b linspace-speedups
    ```

    - 进行本地提交( ``git add`` 和 ``git commit``)时，请使用[标准格式](development_workflow.html#编写提交消息)，正确的提交你的消息，编写在更改之前失败并在更改后通过的测试，在本地运行[所有测试](development_environment.html#测试构建)。确保在docstring中记录任何更改的行为，并遵循NumPy docstring[标准](/bedocs/howto_document.html)。
1. 提交您的贡献：
    - 在GitHub上将更改推回到fork：

    ``` bash
    git push origin linspace-speedups
    ```

    - 输入您的GitHub用户名和密码（重复贡献者或高级用户可以通过[SSH](gitwash.html#set-up-and-configure-a-github-account)连接到GitHub来删除此步骤。

    - 前往GitHub。新的分支将显示一个绿色的Pull Request按钮。确保标题和信息清晰、简洁和不言自明。然后单击按钮提交。

    - 如果您的提交引入了新特性或更改功能，请在[邮件列表](https://mail.python.org/mailman/listinfo/numpy-devel)上发布以解释您的更改。对于错误修复，文档更新等，这通常是不必要的，尽管如果您没有得到任何反应，请随时要求查看。
1. 评审程序:
    - 评审者(其他开发人员和感兴趣的社区成员)将在您的拉取请求(PR)上编写内联和/或一般评论，以帮助您改进其实现、文档和风格。每个在项目中工作的开发人员都会对他们的代码进行审查，我们已经开始把它看作是友好的对话，我们都从中学习到了，并且总体代码质量得到了好处。因此，请不要让评审阻碍你的贡献：它的唯一目的是提高项目的质量，而不是批评(毕竟，我们非常感谢你捐赠的时间！)
    - 要更新您的PR，请在本地存储库上进行更改，提交、运行测试，并且仅当它们成功地推送到您的fork时。一旦这些更改被推上(与之前相同的分支)，PR将自动更新。如果您不知道如何修复测试失败，您可以无论如何推动您的更改，并在PR注释中寻求帮助。
    - 每次PR更新后都会触发各种持续集成(CI)服务，以构建代码、运行单元测试、测量代码覆盖率并检查分支机构的编码风格。在您的PR可以合并之前，CI测试必须通过。如果CI失败，您可以通过单击“失败”图标（红色十字）并检查构建和测试日志来找出原因。为了避免过度使用和浪费此资源，请在提交之前在本地[测试您的工作](development_environment.html)。
    - 在合并之前，请购单必须得到至少一个核心团队成员的批准。批准意味着核心团队成员已经仔细审查了更改，PR已准备好进行合并。
1. 文档更改

    除了更改一般文档中的函数docstring和可能的描述之外，如果您的更改引入了任何面向用户的修改，请更新 ``doc/release/x.xx-notes.rst`` 下的当前发行说明。

    如果您的更改引入了不推荐使用的内容，请确保首先在GitHub或邮件列表上讨论此问题。如果就弃用达成一致，遵循[NEP 23弃用政策](https://numpy.org/neps/nep-0023-backwards-compatibility.html)添加弃用。
1. 交叉引用问题

    如果PR与任何问题相关，您可以将文本 ``xref gh-xxxx`` 添加到GitHub评论中，其中 ``xxxx`` 是问题的编号。同样，如果PR解决了问题，请将 ``xref`` 替换为 ``closes``、``fixes`` 或[GitHub接受](https://help.github.com/en/article/closing-issues-using-keywords)的任何其他风格。

    在源代码中，确保以 ``gh-xxxx`` 作为任何问题或PR引用的前缀。

有关更详细的讨论，请阅读并点击本页底部的链接。

### ``upstream/master`` 与功能分支之间的差异

如果GitHub指示拉取请求的分支不能再自动合并，则必须将自启动以来所做的更改合并到分支中。我们推荐的方法是重新[建立在主要分支](development_workflow.html#重新拉取主分支)的基础上。

### 指导方针

- 所有代码都应该有测试（有关更多详细信息，请参见下面的[测试覆盖率](#测试覆盖率)）。
- 所有代码都应[记录在案](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard)。
- 在未经核心团队成员审核和批准的情况下，不会提交任何更改。如果您在一周内没有收到对拉取请求的回复，请在PR或[邮件列表](https://mail.python.org/mailman/listinfo/numpy-devel)上礼貌地询问。

### 文体指南

- 将编辑器设置为遵循[PEP 8](https://www.python.org/dev/peps/pep-0008/)（删除尾随空白，没有制表符等）。用pyflakes/flke8检查代码。
- 使用numpy数据类型而不是字符串(``np.uint8`` 而不是 ``uint8``)。
- 使用以下导入约定：
    ``` python
    import numpy as np
    ```
- 有关C代码，请参见[numpy-c-style-guide](style_guide.html)。

### 测试覆盖率

修改代码的拉取请求（PR）应该具有新的测试，或者在PR之前将现有的测试修改为失败并且之后通过。你应该在推PR之前[运行测试](development_environment.html)。

理想情况下，模块的测试应覆盖该模块中的所有代码，即语句覆盖率应为100%。

要测量测试覆盖率，请安装[pytest-cov](https://pytest-cov.readthedocs.io/en/latest/)，然后运行：

``` python
$ python runtests.py --coverage
```

这将在 ``Build/Coverage`` 中创建报告，可以通过以下方式查看：

``` python
$ firefox build/coverage/index.html
```

### 构建文档

要构建文档，请从 ``doc`` 目录运行 ``make``。``make help`` 会列出所有目标。例如，要构建HTML文档，可以运行：

``` python
make html
```

然后，所有的HTML文件都将在 ``doc/build/html/`` 中生成。由于文档基于docstring，因此必须在用于运行sphinx的主机python中安装适当版本的numpy。

#### 要求

需要 [Sphinx](https://www.sphinx-doc.org/en/stable/) 来构建文档。Matplotlib和SciPy也是一样的。

#### 修复警告

- “citation not found: R###” 在 docstring 的第一行引用之后可能有下划线(例如 [1]_)。使用此方法查找源文件：$ cd doc/build；grep-rin R####。
- “Duplicate citation R###, other instance in…” 在一个docstring中可能有一个没有[1]的[2]。

## 开发流程 - 详细信息

文章的其余部分：

- [NumPy 行为准则](conduct/code_of_conduct.html)
  - [Introduction](conduct/code_of_conduct.html#introduction)
  - [Specific Guidelines](conduct/code_of_conduct.html#specific-guidelines)
  - [Diversity Statement](conduct/code_of_conduct.html#diversity-statement)
  - [Reporting Guidelines](conduct/code_of_conduct.html#reporting-guidelines)
  - [Incident reporting resolution &amp; Code of Conduct enforcement](conduct/code_of_conduct.html#incident-reporting-resolution-code-of-conduct-enforcement)
  - [Endnotes](conduct/code_of_conduct.html#endnotes)
- [Git Basics](gitwash/index.html)
  - [Install git](gitwash/git_intro.html)
  - [Get the local copy of the code](gitwash/following_latest.html)
  - [Updating the code](gitwash/following_latest.html#updating-the-code)
  - [Getting started with Git development](gitwash/development_setup.html)
  - [Git configuration](gitwash/configure_git.html)
  - [Two and three dots in difference specs](gitwash/dot2_dot3.html)
  - [Additional Git Resources](gitwash/git_resources.html)
- [Setting up and using your development environment](development_environment.html)
  - [Recommended development setup](development_environment.html#recommended-development-setup)
  - [Testing builds](development_environment.html#testing-builds)
  - [Building in-place](development_environment.html#building-in-place)
  - [Other build options](development_environment.html#other-build-options)
  - [Using virtualenvs](development_environment.html#using-virtualenvs)
  - [Running tests](development_environment.html#running-tests)
  - [Rebuilding &amp; cleaning the workspace](development_environment.html#rebuilding-cleaning-the-workspace)
  - [Debugging](development_environment.html#debugging)
  - [Understanding the code &amp; getting started](development_environment.html#understanding-the-code-getting-started)
- [Development workflow](development_workflow.html)
  - [Basic workflow](development_workflow.html#basic-workflow)
  - [Additional things you might want to do](development_workflow.html#additional-things-you-might-want-to-do)
- [NumPy benchmarks](../benchmarking.html)
  - [Usage](../benchmarking.html#usage)
  - [Writing benchmarks](../benchmarking.html#writing-benchmarks)
- [NumPy C Style Guide](style_guide.html)
  - [Introduction](style_guide.html#introduction)
  - [C dialect](style_guide.html#c-dialect)
  - [Code lay-out](style_guide.html#code-lay-out)
  - [Naming conventions](style_guide.html#naming-conventions)
  - [Function documentation](style_guide.html#function-documentation)
- [Releasing a Version](releasing.html)
  - [How to Prepare a Release](releasing.html#how-to-prepare-a-release)
  - [Step-by-Step Directions](releasing.html#step-by-step-directions)
- [NumPy governance](governance/index.html)
  - [NumPy project governance and decision-making](governance/governance.html)
  - [Current steering council and institutional partners](governance/people.html)

NumPy-specific 工作流程在 [numpy-development-workflow](development_workflow.html)。
