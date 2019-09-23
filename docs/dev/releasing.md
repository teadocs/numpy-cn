---
meta:
  - name: keywords
    content: NumPy 发布版本
  - name: description
    content: 此页概述了为NumPy构建二进制版本所需的内容。
---

# 发布版本

## 如何准备发布

此页概述了为NumPy构建二进制版本所需的内容。

### 当前构建和发布信息

关于构建和发布NumPy和SciPy的当前信息分散在几个地方。它应该在一个地方进行总结，更新，并在必要时进行更详细的描述。以下部分列出了可以找到有用信息的所有地方。

#### 源文件树

- INSTALL.rst.txt
- release.sh
- pavement.py

#### NumPy 的文档

- [https://github.com/numpy/numpy/blob/master/doc/HOWTO_RELEASE.rst.txt](https://github.com/numpy/numpy/blob/master/doc/HOWTO_RELEASE.rst.txt)
- [http://projects.scipy.org/numpy/wiki/MicrosoftToolchainSupport](http://projects.scipy.org/numpy/wiki/MicrosoftToolchainSupport)（死链接）

#### SciPy.org 维基

- [https://www.scipy.org/Installing_SciPy](https://www.scipy.org/Installing_SciPy)和该页面上的链接。
- [http://new.scipy.org/building/windows.html](http://new.scipy.org/building/windows.html)（死链接）

#### 文档维基

- [http://docs.scipy.org/numpy/docs/numpy-docs/user/install.rst/](https://docs.scipy.org/numpy/docs/numpy-docs/user/install.rst/)（死链接）

#### 发布脚本

- [https://github.com/numpy/numpy-vendor](https://github.com/numpy/numpy-vendor)

### 支持的平台和版本

从源代码构建时，Python 2.7和> = 3.4是当前支持的版本。每次我们将代码合并到master时，我们都会针对所有这些版本测试NumPy。二进制安装程序可用于这些版本的子集（参见下文）。

#### OS X 

Python 2.7和> = 3.4是我们提供二进制安装程序的版本。支持OS X版本> = 10.6。我们为OSX构建了与Python.org Python，系统Python，自制软件和macports兼容的二进制轮子 - 有关详细信息，请参阅此[OSX轮子构建摘要](https://github.com/MacPython/wiki/wiki/Spinning-wheels)。

#### Windows

我们在Windows上为Python 2.7,3.4,3.5构建32位和64位轮。支持Windows XP，Vista，7,8和10。我们使用Appveyor上的MSVC编译器构建NumPy，但我们希望更新为[mingw-w64工具链](https://mingwpy.github.io)。Windows车轮使用ATLAS进行BLAS/LAPACK。

#### Linux

我们
为NumPy 制造并运送[manylinux1](https://www.python.org/dev/peps/pep-0513)车轮。许多Linux发行版都包含自己的NumPy二进制版本。

#### BSD/Solaris

没有提供二进制文件，但已报告在Solaris和BSD上成功构建。

### 工具链

我们在云基础架构上构建所有轮子 - 因此这个编译器列表用于本地信息和调试构建。请参阅[numpy wheels](https://github.com/MacPython/numpy-wheels) repo中的``.travis.yml``和
 ``appveyor.yml``脚本以获取构建配方的[权威](https://github.com/MacPython/numpy-wheels)来源。注意使用pip可用的包。

#### 编译器

使用相同的gcc版本作为在每个平台上构建Python本身的版本。目前这意味着：

- OS X建立在travis目前使用 *clang* 。当从Python.org安装程序构建Python时，似乎可以从travis-ci OSX 10.9 VM安全地构建OSX> = 10.6的二进制轮。
- Windows构建使用与正在构建的Python相对应的MSVC版本;
- Manylinux1车轮使用Manylinux docker图像上提供的gcc。

你需要Cython来构建二进制文件。Cython将``.pyx``
NumPy发行版中的``.c``文件编译为文件。

#### 构建源档案和轮子

您将需要numpy轮的写入权限才能触发车轮构建。

- 来自[python.org](https://python.org)或Linux发行版的Python 。
- 用Cython
- virtualenv（pip）
- 摊铺机（点子）
- numpy-wheels [https://github.com/MacPython/numpy-wheels](https://github.com/MacPython/numpy-wheels)（克隆）

#### 构建文档

构建文档需要一些 latex ``.sty`` 文件。全部安装以避免加重。

- Sphinx（pip）
- numpydoc（pip）
- Matplotlib
- Texlive（或Windows上的MikTeX）

#### 上传到 PyPI

- terryfy [https://github.com/MacPython/terryfy](https://github.com/MacPython/terryfy)（克隆）。
- beautifulsoup4（pip）
- delocate（pip）
- auditwheel（pip）
- twine（pip）

#### 生成 作者/pr 列表

您将需要一个个人访问令牌
 [https://help.github.com/article/creating-a-personal-access-token-for-the-command-line/，](https://help.github.com/article/creating-a-personal-access-token-for-the-command-line/) 
以便脚本可以访问github NumPy存储库。

- gitpython（pip）
- pygithub（pip）

#### VIRTUALENV 

Virtualenv 是一个非常有用的工具，可以保留多个版本的软件包。它也用于Paver脚本来构建文档。

### 什么是发布

#### 轮子

- 用于32位和64位的Python 2.7,3.4,3.5的Windows轮子，使用Appveyor构建;
- 通过travis-ci构建的双架构OSX轮子;
- 通过travis-ci构建的32位和64位Manylinux1轮子。

有关更多详细信息，请参阅[numpy wheels](https://github.com/MacPython/numpy-wheels)构建存储库。

#### 其他

- 发行说明
- 更新日志

#### 源分发

我们以.zip和.tar.gz格式构建源代码版本。

### 发布流程

#### 同意发布计划

典型的发布时间表是一个测试版，两个发布候选版和最终版本。最好先在邮件列表上讨论时间，以便人们按时完成提交，合并doc wiki编辑等。设置日期后，创建一个新的maintenance/xyz分支，添加新的空发布主分支中下一个版本的注释，并更新Trac里程碑。

#### 确保当前分支正确构建包

``` python
git clean -fxd
python setup.py bdist
python setup.py sdist
```

要在正确设置所有内容后实际构建二进制文件，可以使用release.sh脚本。有关构建过程本身的详细信息，最好阅读pavement.py脚本。

::: tip 注意

对于 beta(s)、发布 candidates(s) 和最终发行版重复以下步骤。

:::

#### 检查弃用

在发布分支之前，应检查是否实际删除了应删除的所有已弃用的代码，并且所有新的弃用都在docstring或deprecation警告中说明代码将被删除的版本。

#### 检查 C API 版本号

C API版本需要在三个地方进行跟踪

- numpy/core/setup_common.py
- numpy/core/code_generators/cversions.txt
- numpy/core/include/numpy/numpyconfig.h

该过程分为三个步骤。

1. 如果API已更改，请在 setup_common.py 中增加 C_API_VERSION。仅当针对当前API编译的任何代码将向后兼容上次发布的NumPy版本时，API才会保持不变。 对C结构的任何更改或对公共接口的添加都将使新API不向后兼容。
1. 如果第一步中的 C_API_VERSION 已更改，或者 API 的哈希值已更改，则需要更新 cversions.txt 文件。要检查哈希，请运行脚本 numpy/core/cversions.py 并记下打印的API哈希。 如果该哈希值与 numpy/core/code_generators/cversions.txt 中的最后一个哈希值不匹配，则哈希值已更改。 使用适当的C_API_VERSION 和 hash，向 cversions.txt 添加一个新条目。 如果API版本未更改，但散列不同，则需要注释掉该API版本的上一个条目。 例如，在NumPy 1.9中添加了注释，这改变了散列，但API与1.8中的相同。 哈希用作API更改的检查，但它不是确定的。
1. 如果步骤1和2正确完成，则编译版本不应发出警告“构建开始时检测到API不匹配”。

numpy/core/setup_common.py 中的 C ABI 版本号只应针对主要版本进行更新。

#### 查看发行说明

使用[towncrier](https://github.com/hawkowl/towncrier)构建发行说明，将其复制到正确的名称，然后提交更改。这将删除所有片段``changelog/*.rst``
并添加``doc/release/latest-note.rst``必须使用正确的版本号重命名：

``` python
python -mtowncrier --version "Numpy 1.11.0"
git mv doc/release/latest-note.rst doc/release/1.11.0-notes.rst
git commit -m"Create release note"
```

检查发行说明是否是最新的。

使用“亮点”部分更新发行说明。提及以下一些内容：

- 主要新功能
- 已弃用和已删除的功能
- 支持的Python版本
- for SciPy，支持NumPy版本
- 不久的将来的展望

#### 更新发布状态并创建发布“标签” 

识别发布的提交哈希，例如1b2e1d63ff。

``` bash
git co 1b2e1d63ff # gives warning about detached head
```

首先，``pavement.py``根据发布版本更改/检查以下变量：

``` python
RELEASE_NOTES = 'doc/release/1.7.0-notes.rst'
LOG_START = 'v1.6.0'
LOG_END = 'maintenance/1.7.x'
```

做任何其他改变。准备好发布时，请执行以下更改：

``` python
diff --git a/setup.py b/setup.py
index b1f53e3..8b36dbe 100755
--- a/setup.py
+++ b/setup.py
@@ -57,7 +57,7 @@ PLATFORMS           = ["Windows", "Linux", "Solaris", "Mac OS-
 MAJOR               = 1
 MINOR               = 7
 MICRO               = 0
-ISRELEASED          = False
+ISRELEASED          = True
 VERSION             = '%d.%d.%drc1' % (MAJOR, MINOR, MICRO)

 # Return the git revision as a string
```

并确保``VERSION``变量设置正确。

现在您可以进行发布提交和标记。我们建议您不要立即推送提交或标记，以防您需要进行更多清理。我们更愿意推迟标记的推送，直到我们确信这是已发布代码的确切形式（请参阅：[推送发布标记并提交](#push-tag-and-commit)）：

该``-s``标志生成PGP（通常是GPG）签名标记。请签署发布标签。

release标记应该在注释（标记消息）中具有版本号。不幸的是，标签的名称可以在不破坏签名的情况下进行更改，但消息的内容却不能。

请参阅：[https://github.com/scipy/scipy/issues/4919](https://github.com/scipy/scipy/issues/4919)用于签名的释放标签的讨论，并[https://keyring.debian.org/creating-key.html](https://keyring.debian.org/creating-key.html)有关创建，如果你一个GPG密钥的说明没有一个。

为了使您的密钥更易于识别，请考虑将密钥发送到公共密钥服务器，其命令如下：

``` python
gpg --send-keys <yourkeyid>
```

#### 应用补丁来修复虚假步骤

NPY_RELAXED_STRIDE_CHECKING在NumPy 1.10.0中成为默认值，并且在开发分支中使用伪造的步幅来排除问题。该
 [补丁](https://github.com/numpy/numpy/pull/5996)如有必要应进行更新，并应用到发布分支理顺进展。

#### 更新主分支的版本

增加setup.py中的版本号。发布候选版本应在XYZ格式后附加“rc1”（或“rc2”，“rcN”）。

还可以在cversions.txt中创建新版本哈希，并在numpyconfig.h中定义相应的版本NPY_x_y_API_VERSION

#### 在travis-ci和Appveyor上触发轮子构建

查看 *numpy wheels* 存储库。

在该存储库中编辑文件：

- ``.travis.yml``;
- ``appveyor.yml``。

在这两种情况下，将``BUILD_COMMIT``变量设置为当前版本标记 - 例如``v1.11.1``。

确保已按下释放标记。

做一个承诺让您的修改触发构建``.travis.yml``，并
 ``appveyor.yml``到仓库：

``` python
cd /path/to/numpy-wheels
# Edit .travis.yml, appveyor.yml
git commit
git push
```

轮子一旦构建，就会出现在Rackspace容器中，指向：

- [http://wheels.scipy.org](http://wheels.scipy.org)
- [https://3f23b170c54c2533c070-1c8a9b3114517dc5fe17b7c3f8c63a43.ssl.cf2.rackcdn.com](https://3f23b170c54c2533c070-1c8a9b3114517dc5fe17b7c3f8c63a43.ssl.cf2.rackcdn.com)

HTTP地址可能会先更新，您应该在构建完成后等待15分钟才能获取二进制文件。

#### 发布

构建更改日志和备注以供上载：

``` python
paver write_release_and_log
```

#### 构建和归档文档

做：

``` python
cd doc/
make dist
```

检查文档是否处于可构建状态。然后，在标记之后，在numpy/doc repo中创建文档的存档：

``` python
# This checks out github.com/numpy/doc and adds (``git add``) the
# documentation to the checked out repo.
make merge-doc
# Now edit the ``index.html`` file in the repo to reflect the new content,
# and commit the changes
git -C dist/merge commit -a "Add documentation for <version>"
# Push to numpy/doc repo
git -C push
```

#### 更新的PyPI 

轮子和源应该上传到PyPI。

您应该首先上传轮子，并将源格式保留到最后，以确保pip用户在期望二进制轮时不会意外地获得源安装。

您可以使用[https://github.com/MacPython/terryfy中](https://github.com/MacPython/terryfy)的``wheel-uploader``脚本
 自动执行此操作。这是推荐下载所有Windows，Manylinux，OSX轮子并上传到PyPI的咒语。[](https://github.com/MacPython/terryfy)

``` python
NPY_WHLS=~/wheelhouse   # local directory to cache wheel downloads
CDN_URL=https://3f23b170c54c2533c070-1c8a9b3114517dc5fe17b7c3f8c63a43.ssl.cf2.rackcdn.com
wheel-uploader -u $CDN_URL -w $NPY_WHLS -v -s -t win numpy 1.11.1rc1
wheel-uploader -u $CDN_URL -w warehouse -v -s -t macosx numpy 1.11.1rc1
wheel-uploader -u $CDN_URL -w warehouse -v -s -t manylinux1 numpy 1.11.1rc1
```

该``-v``标志提供详细的反馈，``-s``使脚本在上载之前使用您的GPG密钥对轮子进行签名。不要忘记在源tarball之前上传轮子，因此人们没有时间从预期的二进制安装切换到PyPI的源安装。

有两种方法可以更新PyPI上的源代码，第一种方法是：

``` python
$ git clean -fxd  # to be safe
$ python setup.py sdist --formats=gztar,zip  # to check
# python setup.py sdist --formats=gztar,zip upload --sign
```

这将要求您的密钥PGP密码，以便签署构建的源包。

第二种方法是在PyPI的Web界面中将sdist目录中的PKG_INFO文件上传。源tarball也可以通过此接口上传。

#### 推送发布标签并提交

最后，现在您确信此标记正确定义了您发布的源代码，您可以将标记推送到github并释放提交：

``` python
git push  # Push release commit
git push upstream <version>  # Push tag named <version>
```

其中``upstream``指向主[https://github.com/numpy/numpy.git](https://github.com/numpy/numpy.git) 
存储库。

#### 更新scipy.org 

带有下载站点链接的发布公告应放在scipy.org首页的侧边栏中。

scipy.org应该是[https://github.com/scipy/scipy.org](https://github.com/scipy/scipy.org)上的公关。需要修改的文件是``www/index.rst``。搜索``News``。

#### 宣布列表

发布应该在NumPy和SciPy的邮件列表上公布，发布到python-announce，也可能在Matplotlib，IPython和/或Pygame上公布。

在beta/RC阶段，应该在邮件列表上发布一个明确的请求，要求使用其他几个库（SciPy/Matplotlib/Pygame）测试二进制文件。

#### 宣布Linux每周新闻

通过电子邮件发送LWN编辑，让他们知道发布。路线：[https](https://lwn.net/op/FAQ.lwn#contact)：
 [//lwn.net/op/FAQ.lwn#contact](https://lwn.net/op/FAQ.lwn#contact)

#### 最终发布后

在宣布最终版本之后，还有一些管理任务要做：

- 发布分支中的转发端口更改，以将注释和释放脚本（如果有）发布到主分支。
- 更新Trac中的里程碑。

## 循序渐进

此文件包含Linux上NumPy 1.14.5发行版的演练。命令可以复制到命令行中，但一定要用正确的版本替换1.14.5。

### 发布演练

请注意，在下面的代码片段中，``upstream``引用github上的根存储库和``origin``个人帐户中的fork。如果您没有分叉存储库但只是在本地克隆它，则可能需要进行调整。如果它尚不存在，您也可以编辑``.git/config``和添加``upstream``。

#### Backport PullRequests

必须将已为此版本标记的更改后移到maintenance/1.14.x分支。

#### 更新发布文档

``doc/changelog/1.14.5-changelog.rst``应更新该文件以反映最终的更改列表和贡献者。该文本可以通过以下方式生成：

``` python
$ python tools/changelog.py $GITHUB v1.14.4..maintenance/1.14.x > doc/changelog/1.14.5-changelog.rst
```

其中``GITHUB``包含你的github访问令牌。此文本也可以附加到``doc/release/1.14.5-notes.rst``发布更新中，但不适用于新版本``1.14.0``，因为``*.0``发布的更改日志过长。``doc/source/release.rst``还应使用指向新版本说明的链接更新该文件。这些更改应该提交给维护分支，稍后将被转发到master。

#### 完成发行说明

填写发布说明，``doc/release/1.14.5-notes.rst``呼吁进行重大更改。

#### 准备发布提交

检查发布的分支，确保它是最新的，并清理存储库：

``` python
$ git checkout maintenance/1.14.x
$ git pull upstream maintenance/1.14.x
$ git submodule update
$ git clean -xdfq
```

按照HOWTO_RELEASE中的详细说明编辑pavement.py和setup.py：

``` python
$ gvim pavement.py setup.py
$ git commit -a -m"REL: NumPy 1.14.5 release."
```

完整性检查：

``` python
$ python runtests.py -m "full"  # NumPy < 1.17 only
$ python3 runtests.py -m "full"
```

将此版本直接推送到维护分支的末尾。这需要对numpy存储库的写权限：

``` python
$ git push upstream maintenance/1.14.x
```

例如，请参阅1.14.3 REL提交：[https](https://github.com/numpy/numpy/commit/73299826729be58cec179b52c656adfcaefada93)：[//github.com/numpy/numpy/commit/73299826729be58cec179b52c656adfcaefada93](https://github.com/numpy/numpy/commit/73299826729be58cec179b52c656adfcaefada93)。

#### 构建源代码发布

Paver用于构建源版本。它将创建``release``和
 ``release/installers``目录，``*.zip``并``*.tar.gz``
在后者中放置和源代码。

``` python
$ python3 -m cython --version  # check for correct cython version
$ paver sdist  # sdist will do a git clean -xdf, so we omit that
```

#### 构建轮子

通过在此提交中指向numpy-wheels存储库来触发轮子构建。这可能需要一段时间。numpy-wheels存储库是从[https://github.com/MacPython/numpy-wheels](https://github.com/MacPython/numpy-wheels)克隆的
 。以拉动开始，因为可能已由其他人访问和更改了回购并且推送将失败：

``` python
$ cd ../numpy-wheels
$ git pull upstream master
$ git branch <new version>  # only when starting new numpy version
$ git checkout v1.14.x  # v1.14.x already existed for the 1.14.4 release
```

编辑``.travis.yml``和``.appveyor.yml``文件以确保它们具有正确的版本，并为``REL``上面创建的提交输入提交哈希``BUILD_COMMIT``，请参阅 *v1.14.3中* 的_example ：

``` python
$ gvim .travis.yml .appveyor.yml
$ git commit -a
$ git push upstream HEAD
```

现在等 如果您对所花费的时间感到紧张 - 构建可能需要几个小时 - 您可以按照[https://github.com/MacPython/numpy-wheels](https://github.com/MacPython/numpy-wheels)上提供的链接检查构建进度，以检查travis和appveyor构建状态。在继续之前，检查是否已经构建并上载了所有需要的轮子。目前应该有22个在
 [https://wheels.scipy.org,4](https://wheels.scipy.org)个用于Mac，8个用于Windows，10个用于Linux。请注意，有时构建（如测试）会因无关原因而失败，您需要重新启动它们。

#### 下载轮

当轮子都已成功构建后，使用存储库``wheel-uploader``
中的下载它们``terryfy``。如果您还没有，可以从[https://github.com/MacPython/terryfy](https://github.com/MacPython/terryfy)克隆terryfy存储库
 。轮子也可以使用``wheel-uploader``，但我们更喜欢将所有轮子下载到``../numpy/release/installers``目录，然后使用``twine``以下内容上传：

``` python
$ cd ../terryfy
$ git pull upstream master
$ CDN_URL=https://3f23b170c54c2533c070-1c8a9b3114517dc5fe17b7c3f8c63a43.ssl.cf2.rackcdn.com
$ NPY_WHLS=../numpy/release/installers
$ ./wheel-uploader -u $CDN_URL -n -v -w $NPY_WHLS -t win numpy 1.14.5
$ ./wheel-uploader -u $CDN_URL -n -v -w $NPY_WHLS -t manylinux1 numpy 1.14.5
$ ./wheel-uploader -u $CDN_URL -n -v -w $NPY_WHLS -t macosx numpy 1.14.5
```

如果经常这样做，请考虑将CDN_URL和NPY_WHLS作为默认环境的一部分。

#### 生成README文件

这需要在下载所有安装程序之后，但在更新路面文件以继续开发之前完成：

``` python
$ cd ../numpy
$ paver write_release
```

#### 标记发布

一旦构建并下载了轮子没有错误，请返回维护分支中的numpy存储库并标记``REL``提交，使用gpg密钥对其进行签名：

``` python
$ git tag -s v1.14.5
```

您应该将您的公共gpg密钥上传到github，以便标记在那里显示为“已验证”。

检查文件``release/installers``是否具有正确的版本，然后将标签推送到上游：

``` python
$ git push upstream v1.14.5
```

我们要等到这一点才能推送标签，因为它是公共的，在推送之后不应该更改。

#### 将维护分支重置为开发状态

``REL``向numpy维护分支添加另一个提交，该分支将``ISREALEASED``标志重置
 为``False``并递增版本计数器：

``` python
$ gvim pavement.py setup.py
```

为下一个版本创建发行说明并编辑它们以设置版本：

``` python
$ cp doc/release/template.rst doc/release/1.14.6-notes.rst
$ gvim doc/release/1.14.6-notes.rst
$ git add doc/release/1.14.6-notes.rst
```

在文档发布列表中添加新的发行说明：

``` python
$ gvim doc/source/release.rst
```

提交结果：

``` python
$ git commit -a -m"REL: prepare 1.14.x for further development"
$ git push upstream maintenance/1.14.x
```

#### 上传到PyPI将

使用上传到PyPI ``twine``。最近的版本``twine``中，需要最近的PyPI更改之后，版本``1.11.0``在这里使用。

``` sh
$ cd ../numpy
$ twine upload release/installers/*.whl
$ twine upload release/installers/numpy-1.14.5.zip  # Upload last.
```

如果其中一个命令在中间断开（这并不罕见），您可能需要有选择地上传其余文件，因为PyPI不允许同一文件上载两次。如果pip用户在此过程中访问文件，则应最后上载源文件以避免同步问题。请注意，PyPI只允许单个源代码分发，这里我们选择了zip归档文件。

#### 上传文件到github上

转到[https://github.com/numpy/numpy/releases](https://github.com/numpy/numpy/releases)，应该有一个，单击它并点击该标签的编辑按钮。使用可编辑的文本窗口和二进制上载，有两种方法可以添加文件。``v1.14.5
tag``

- 将``release/README.md``文件内容剪切并粘贴到文本窗口中。
- 上传``release/installers/numpy-1.14.5.tar.gz``为二进制文件。
- 上传``release/installers/numpy-1.14.5.zip``为二进制文件。
- 上传``release/README.rst``为二进制文件。
- 上传``doc/changelog/1.14.5-changelog.rst``为二进制文件。
- 如果这是预发行版，请检查预发布按钮。
- 点击底部的按钮。``{Publish,Update} release``

#### 上传文件numpy.org 

此步骤仅适用于最终版本，可以跳过预发布版本。克隆repo
 并使用新文档更新它：``make merge-doc``numpy/doc``doc/build/merge``

``` python
$ pushd doc
$ make dist
$ make merge-doc
$ popd
```

如果发布系列是新版本，则需要``doc/build/merge/index.html``在“插入此处”注释之后向首页添加新部分
 ：

``` python
$ gvim doc/build/merge/index.html +/'insert here'
```

否则，只应使用新标记名称更新``zip``和``pdf``链接：

``` python
$ gvim doc/build/merge/index.html +/'tag v1.14'
```

您可以在浏览器中“测试运行”新文档，以确保链接有效：

``` python
$ firefox doc/build/merge/index.html
```

一切似乎都令人满意，请提交并上传更改：

``` python
$ pushd doc/build/merge
$ git commit -am"Add documentation for v1.14.5"
$ git push
$ popd
```

#### 在scipy.org上宣布发布

这假设您已经分叉[https://github.com/scipy/scipy.org](https://github.com/scipy/scipy.org)：

``` python
$ cd ../scipy.org
$ git checkout master
$ git pull upstream master
$ git checkout -b numpy-1.14.5
$ gvim www/index.rst # edit the News section
$ git commit -a
$ git push origin HEAD
```

现在转到你的fork并为分支发出pull请求。

#### 宣布邮件列表

该发布应该在numpy-discussion，scipy-devel，scipy-user和python-announce-list邮件列表上公布。查看以前的基本模板公告。贡献者和PR列表与上面的发行说明生成的相同。如果你是crosspost，请确保python-announce-list是BCC，这样回复就不会发送到该列表。

#### 发布后任务

转发文档更改的转发端口``doc/release/1.14.5-notes.rst``，
 ``doc/changelog/1.14.5-changelog.rst``并将发行说明添加到
 ``doc/source/release.rst``。
