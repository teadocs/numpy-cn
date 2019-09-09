# 开发工作流程

您已经拥有自己的 [NumPy](https://www.numpy.org) 存储库的分叉副本，
通过以下方式 [制作自己的NumPy副本（Fork）](gitwash.html#forking)以及
[设置您的fork](gitwash.html#set-up-fork)，您已经按照[Git配置](gitwash.html#configure-git)了 [git](https://git-scm.com/) ，
并且已经链接了上游存储库，如将您的存储库[链接到上游仓库](gitwash.html#linking-to-upstream)中所述。

下面介绍的是Git的推荐工作流程。

## 基本工作流程

简而言之：

1. 为您执行的每组编辑启动一个新 *功能分支* 。见[下文](#making-a-new-feature-branch)。
1. 就是干！见[下文](#editing-workflow)
1. 等结束了：
  - *贡献者*：将您的功能分支推送到您自己的Github仓库，并 [创建一个拉取请求](#asking-for-merging)。
  - *核心开发者*：如果想更改推不进一步审查，看笔记[如下](#pushing-to-main)。

这种工作方式有助于使工作井井有条，并使历史尽可能清晰。

::: tip 另见

有许多在线教程可以帮助您[学习git](https://www.atlassian.com/git/tutorials/)。有关特定git工作流的讨论，请参阅有关[linux git工作流](https://www.mail-archive.com/dri-devel@lists.sourceforge.net/msg39091.html)和[ipython git工作流的](https://mail.python.org/pipermail/ipython-dev/2010-October/006746.html)这些讨论。

:::

### 创建新的功能分支

首先，从``upstream``存储库中获取新的提交：

``` python
git fetch upstream
```

然后，基于上游存储库的主分支创建新分支：

``` python
git checkout -b my-new-feature upstream/master
```

### 编辑工作流程

#### 概述

``` python
# hack hack
git status # Optional
git diff # Optional
git add modified_file
git commit
# push the branch to your own Github repo
git push origin my-new-feature
```

#### 更详细

1. Make some changes. When you feel that you’ve made a complete, working set of related changes, move on to the next steps.
1. Optional: Check which files have changed with git status (see git status). You’ll see a listing like this one:
    ```
    # On branch my-new-feature
    # Changed but not updated:
    #   (use "git add <file>..." to update what will be committed)
    #   (use "git checkout -- <file>..." to discard changes in working directory)
    #
    #  modified:   README
    #
    # Untracked files:
    #   (use "git add <file>..." to include in what will be committed)
    #
    #  INSTALL
    no changes added to commit (use "git add" and/or "git commit -a")
    ```
1. Optional: Compare the changes with the previous version using with ``git diff`` ([git diff](https://www.kernel.org/pub/software/scm/git/docs/git-diff.html)). This brings up a simple text browser interface that highlights the difference between your files and the previous version.
1. Add any relevant modified or new files using ``git add modified_file`` (see [git add](https://www.kernel.org/pub/software/scm/git/docs/git-add.html)). This puts the files into a staging area, which is a queue of files that will be added to your next commit. Only add files that have related, complete changes. Leave files with unfinished changes for later commits.
1. To commit the staged files into the local copy of your repo, do ``git commit``. At this point, a text editor will open up to allow you to write a commit message. Read the [commit message section](https://numpy.org/devdocs/dev/development_workflow.html#writing-the-commit-message) to be sure that you are writing a properly formatted and sufficiently detailed commit message. After saving your message and closing the editor, your commit will be saved. For trivial commits, a short commit message can be passed in through the command line using the ``-m`` flag. For example, ``git commit -am "ENH: Some message"``.
1. In some cases, you will see this form of the commit command: ``git commit -a``. The extra ``-a`` flag automatically commits all modified files and removes all deleted files. This can save you some typing of numerous ``git add`` commands; however, it can add unwanted changes to a commit if you’re not careful. For more information, see [why the -a flag?](http://www.gitready.com/beginner/2009/01/18/the-staging-area.html) - and the helpful use-case description in the [tangled working copy problem](https://tomayko.com/writings/the-thing-about-git).
1. Push the changes to your forked repo on [github](https://github.com/numpy/numpy):
    ``` bash
    git push origin my-new-feature
    ```

    For more information, see [git push](https://www.kernel.org/pub/software/scm/git/docs/git-push.html).

::: tip 注意

假设您已按照这些页面中的说明操作，git将创建一个指向您的[github](https://github.com/numpy/numpy) repo 的默认链接``origin``。在git> = 1.7中，您可以使用以下``--set-upstream``选项确保永久设置到origin的链接：

``` python
git push --set-upstream origin my-new-feature
```

从现在开始，[git](https://git-scm.com/)将知道这``my-new-feature``与``my-new-feature``您自己的[github仓库中](https://github.com/numpy/numpy)的分支有关。随后的推送调用命令将简化为以下写法：

``` python
git push
```

您必须使用``--set-upstream``您创建的每个新分支。

:::

可能的情况是，当您处理编辑时，会添加新的提交，``upstream`` 这会影响您的工作。
在这种情况下，请按照本文档的 [Rebasing on master](#rebasing-on-master)部分将这些更改应用于您的分支。

#### 编写提交消息

提交消息应该是明确的，并遵循一些基本规则。例：

``` python
ENH: add functionality X to numpy.<submodule>.

The first line of the commit message starts with a capitalized acronym
(options listed below) indicating what type of commit this is.  Then a blank
line, then more text if needed.  Lines shouldn't be longer than 72
characters.  If the commit is related to a ticket, indicate that with
"See #3456", "See ticket 3456", "Closes #3456" or similar.
```

描述变更的动机，bug修复的bug的本质，或者关于增强所做的一些细节，也可以包含在提交消息中。
消息应该是可以理解的，而不需要查看代码更改。像 ``Maint：Fixed Another one `` 这样的提交消息就是不应该做的事情的一个例子；
读者必须到别处去寻找上下文。

启动提交消息的标准首字母缩写词是：

``` python
API: an (incompatible) API change
BENCH: changes to the benchmark suite
BLD: change related to building numpy
BUG: bug fix
DEP: deprecate something, or remove a deprecated object
DEV: development tool or utility
DOC: documentation
ENH: enhancement
MAINT: maintenance commit (refactoring, typos, etc.)
REV: revert an earlier commit
STY: style fix (whitespace, PEP8)
TST: addition or modification of tests
REL: related to releasing numpy
```

### 要求您的更改与主仓库合并

当您觉得您的工作已经完成时，您可以创建拉取请求（PR）。Github有一个很好的帮助页面，概述了[提交拉取请求](https://help.github.com/articles/using-pull-requests/#initiating-the-pull-request)的过程。

如果您的更改涉及API的修改或功能的添加/修改，您应该：

- 发送电子邮件到[NumPy邮件列表](https://scipy.org/scipylib/mailing-lists.html)，其中包含您PR的链接以及您的更改的描述和动机。这可能会产生变化和反馈。如果您的更改可能存在争议，那么从这一步骤开始可能是谨慎的做法。
- ``doc/release/upcoming_changes/``按照``doc/release/upcoming_changes/README.rst``文件中的说明和格式向目录
 添加发行说明。

### 重新拉取主分支

这将通过上游[NumPy github存储库的](https://github.com/numpy/numpy)更改来更新您的功能分支。如果你不是绝对需要这样做，尽量避免这样做，除非你完成了。第一步是使用来自上游的新提交更新远程存储库：

``` python
git fetch upstream
```

接下来，您需要更新功能分支：

``` python
# go to the feature branch
git checkout my-new-feature
# make a backup in case you mess up
git branch tmp my-new-feature
# rebase on upstream master branch
git rebase upstream/master
```

如果您对上游已更改的文件进行了更改，则可能会生成需要解决的合并冲突。在这种情况下，请参阅
 [下面](#recovering-from-mess-up)的帮助。

最后，在成功的rebase后删除备份分支：

``` python
git branch -D tmp
```

::: tip 注意

在master上重新绑定比将上游合并到您的分支更受欢迎。在处理功能分支时使用和不鼓励使用。``git merge````git pull``

:::

### 从混乱中恢复

有时候，你搞砸了合并或重组。幸运的是，在Git中，从这些错误中恢复是相对简单的。

如果你在一次变革中陷入困境：

``` python
git rebase --abort
```

如果你注意到在rebase之后搞砸了你：

``` python
# reset branch back to the saved point
git reset --hard tmp
```

如果您忘记创建备份分支：

``` python
# look at the reflog of the branch
git reflog show my-feature-branch

8630830 my-feature-branch@{0}: commit: BUG: io: close file handles immediately
278dd2a my-feature-branch@{1}: rebase finished: refs/heads/my-feature-branch onto 11ee694744f2552d
26aa21a my-feature-branch@{2}: commit: BUG: lib: make seek_gzip_factory not leak gzip obj
...

# reset the branch to where it was before the botched rebase
git reset --hard my-feature-branch@{2}
```

如果您实际上并没有陷入困境，但存在合并冲突，则需要解决这些问题。这可能是一个比较棘手的事情。有关如何执行此操作的详细说明，请参阅[有关合并冲突的文章](https://git-scm.com/book/en/Git-Branching-Basic-Branching-and-Merging#Basic-Merge-Conflicts)。

## 您可能想要做的其他事情

### 重写提交历史

::: tip 注意

仅对您自己的功能分支执行此操作。

:::

你做出的承诺中有一个令人尴尬的错字？或许你做了几个错误的开始，你希望后人不要看。

这可以通过 *交互式变基* 来完成。

假设提交历史记录如下所示：

``` python
git log --oneline
eadc391 Fix some remaining bugs
a815645 Modify it so that it works
2dec1ac Fix a few bugs + disable
13d7934 First implementation
6ad92e5 * masked is now an instance of a new object, MaskedConstant
29001ed Add pre-nep for a copule of structured_array_extensions.
...
```

并且``6ad92e5``是``master``分支中的最后一次提交。假设我们要进行以下更改：

- 重写提交消息以``13d7934``获得更明智的信息。
- 合并的提交``2dec1ac``，``a815645``，``eadc391``到一个单一的一个。

我们做如下：

``` python
# make a backup of the current state
git branch tmp HEAD
# interactive rebase
git rebase -i 6ad92e5
```

这将打开一个编辑器，其中包含以下文本：

``` python
pick 13d7934 First implementation
pick 2dec1ac Fix a few bugs + disable
pick a815645 Modify it so that it works
pick eadc391 Fix some remaining bugs

# Rebase 6ad92e5..eadc391 onto 6ad92e5
#
# Commands:
#  p, pick = use commit
#  r, reword = use commit, but edit the commit message
#  e, edit = use commit, but stop for amending
#  s, squash = use commit, but meld into previous commit
#  f, fixup = like "squash", but discard this commit's log message
#
# If you remove a line here THAT COMMIT WILL BE LOST.
# However, if you remove everything, the rebase will be aborted.
#
```

为了实现我们想要的目标，我们将对其进行以下更改：

``` python
r 13d7934 First implementation
pick 2dec1ac Fix a few bugs + disable
f a815645 Modify it so that it works
f eadc391 Fix some remaining bugs
```

这意味着（i）我们想要编辑提交消息
 ``13d7934``，以及（ii）将最后三个提交合并为一个。现在我们保存并退出编辑器。

然后，Git会立即调出一个编辑器来编辑提交消息。修改后，我们得到输出：

``` python
[detached HEAD 721fc64] FOO: First implementation
 2 files changed, 199 insertions(+), 66 deletions(-)
[detached HEAD 0f22701] Fix a few bugs + disable
 1 files changed, 79 insertions(+), 61 deletions(-)
Successfully rebased and updated refs/heads/my-feature-branch.
```

历史现在看起来像这样：

``` python
0f22701 Fix a few bugs + disable
721fc64 ENH: Sophisticated feature
6ad92e5 * masked is now an instance of a new object, MaskedConstant
```

如果出现问题，可以再次进行恢复，[如上所述](#recovering-from-mess-up)。

### 删除github上的分支

``` python
git checkout master
# delete branch locally
git branch -D my-unwanted-branch
# delete branch on github
git push origin :my-unwanted-branch
```

（请注意``:``以前的冒号``test-branch``。另请参阅：[https](https://github.com/guides/remove-a-remote-branch)：
 [//github.com/guides/remove-a-remote-branch](https://github.com/guides/remove-a-remote-branch)

### 几个人共享一个存储库

如果你想与其他人一起工作，你们都在同一个存储库，甚至同一个分支中，那么只需通过[github](https://github.com/numpy/numpy)共享它。

首先将NumPy分配到您的帐户，例如从NumPy [制作您自己的副本（分叉）](gitwash/development_setup.html#forking)。

然后，转到你的分叉存储库github页面，比方说 ``https://github.com/your-user-name/numpy``

点击“管理”按钮，然后将其他任何人作为协作者添加到仓库：

![pull_button](/static/images/pull_button.png)

现在这些人都能做到：

``` python
git clone git@github.com:your-user-name/numpy.git
```

请记住，以``git@``使用ssh协议开头的链接是可读写的; 以＃开头的链接``git://``是只读的。

然后，您的协作者可以通常使用以下方式直接进入该回购：

``` python
git commit -am 'ENH - much better code'
git push origin my-feature-branch # pushes directly into your repo
```

### 探索您的存储库

要查看存储库分支和提交的图形表示：

``` python
gitk --all
```

要查看此分支的提交历史列表：

``` python
git log
```

您也可以在 [网络图形可视化](https://github.com/blog/39-say-hello-to-the-network-graph-visualizer) 工具中查看您的 [GitHub](https://github.com/numpy/numpy) 仓库。

### 反向移植

Backporting是将[numpy / master中](https://github.com/numpy/numpy)提交的新功能/修复复制
 回稳定版本分支的过程。要做到这一点，你要从你正在向后移植的分支上做一个分支，樱桃挑选你想要的提交
 ``numpy/master``，然后提交一个包含backport的分支的pull请求。

1. First, you need to make the branch you will work on. This needs to be based on the older version of NumPy (not master):
    ``` bash
    # Make a new branch based on numpy/maintenance/1.8.x,
    # backport-3324 is our new name for the branch.
    git checkout -b backport-3324 upstream/maintenance/1.8.x
    ```
1. Now you need to apply the changes from master to this branch using [git cherry-pick](https://www.kernel.org/pub/software/scm/git/docs/git-cherry-pick.html):

    ``` bash
    # Update remote
    git fetch upstream
    # Check the commit log for commits to cherry pick
    git log upstream/master
    # This pull request included commits aa7a047 to c098283 (inclusive)
    # so you use the .. syntax (for a range of commits), the ^ makes the
    # range inclusive.
    git cherry-pick aa7a047^..c098283
    ...
    # Fix any conflicts, then if needed:
    git cherry-pick --continue
    ```
1. You might run into some conflicts cherry picking here. These are resolved the same way as merge/rebase conflicts. Except here you can use [git blame](https://www.kernel.org/pub/software/scm/git/docs/git-blame.html) to see the difference between master and the backported branch to make sure nothing gets screwed up.
1. Push the new branch to your Github repository:
    ``` bash
    git push -u origin backport-3324
    ```
1. Finally make a pull request using Github. Make sure it is against the maintenance branch and not master, Github will usually suggest you make the pull request against master.

### 将更改推送到主分支

*仅当您拥有主NumPy仓库的提交权限时，这才有意义。* 

如果您准备好NumPy ``master``或``maintenance``分支的功能分支中有一组“就绪”更改，则可以按``upstream``如下方式将它们推送到：

1. First, merge or rebase on the target branch.
    1. Only a few, unrelated commits then prefer rebasing:
      ``` bash
      git fetch upstream
      git rebase upstream/master
      ```
      See [Rebasing on master](https://numpy.org/devdocs/dev/development_workflow.html#rebasing-on-master).
    1. If all of the commits are related, create a merge commit:
      ``` bash
      git fetch upstream
      git merge --no-ff upstream/master
      ```
1. Check that what you are going to push looks sensible:
    ``` bash
    git log -p upstream/master..
    git log --oneline --graph
    ```
1. Push to upstream:
    ``` bash
    git push upstream my-feature-branch:master
    ```

::: tip 注意

通常最好使用``-n``标志来检查您是否要将所需的更改推送到所需的位置。``git push``

:::