# Git for development

These pages describe a general [git](https://git-scm.com/) and [github](https://github.com/numpy/numpy) workflow.

This is not a comprehensive [git](https://git-scm.com/) reference. It’s tailored to the [github](https://github.com/numpy/numpy) hosting service. You may well find better or quicker ways of getting stuff done with [git](https://git-scm.com/), but these should get you started.

For general resources for learning [git](https://git-scm.com/) see [Additional Git Resources](https://numpy.org/devdocs/dev/gitwash/git_resources.html#git-resources).

Have a look at the [github](https://github.com/numpy/numpy) install help pages available from [github help](https://help.github.com/)

## Install git

Developing with git can be done entirely without github. Git is a distributed version control system. In order to use git on your machine you must [install it](https://git-scm.com/downloads).

## Get the local copy of the code

From the command line:

``` bash
git clone git://github.com/numpy/numpy.git
```

You now have a copy of the code tree in the new numpy directory. If this doesn’t work you can try the alternative read-only url:

``` bash
git clone https://github.com/numpy/numpy.git
```

## Updating the code

From time to time you may want to pull down the latest code. Do this with:

``` bash
cd numpy
git fetch
git merge --ff-only
```

The tree in ``numpy`` will now have the latest changes from the initial repository.

## Getting started with Git development

This section and the next describe in detail how to set up git for working with the NumPy source code. If you have git already set up, skip to [Development workflow](https://numpy.org/devdocs/dev/development_workflow.html#development-workflow).

### Basic Git setup

- [Install git](https://matplotlib.org/devel/gitwash/git_install.html#install-git).
- Introduce yourself to Git:

``` bash
git config --global user.email you@yourdomain.example.com
git config --global user.name "Your Name Comes Here"
```

### Making your own copy (fork) of NumPy

You need to do this only once. The instructions here are very similar to the instructions at [http://help.github.com/forking/](https://help.github.com/forking/) - please see that page for more detail. We’re repeating some of it here just to give the specifics for the [NumPy](https://www.numpy.org/) project, and to suggest some default names.

#### Set up and configure a github account

If you don’t have a [github](https://github.com/numpy/numpy) account, go to the [github](https://github.com/numpy/numpy) page, and make one.

You then need to configure your account to allow write access - see the ``Generating SSH keys`` help on [github help](https://help.github.com/).

#### Create your own forked copy of NumPy

1. Log into your [github](https://github.com/numpy/numpy) account.
1. Go to the [NumPy](https://www.numpy.org/) github home at [NumPy github](https://github.com/numpy/numpy).
1. Click on the *fork* button:

![fork](/static/images/forking_button.png)

After a short pause, you should find yourself at the home page for your own forked copy of [NumPy](https://www.numpy.org/).

### Set up your fork

First you follow the instructions for [Making your own copy (fork) of NumPy](https://numpy.org/devdocs/dev/gitwash/development_setup.html#forking).

#### Overview

``` bash
git clone https://github.com/your-user-name/numpy.git
cd numpy
git remote add upstream https://github.com/numpy/numpy.git
```

#### In detail

##### Clone your fork

1. Clone your fork to the local computer with git clone https://github.com/your-user-name/numpy.git
1. Investigate. Change directory to your new repo: cd numpy. Then git branch -a to show you all branches. You’ll get something like:

    ``` bash
    * master
    remotes/origin/master
    ```

    This tells you that you are currently on the ``master`` branch, and that you also have a ``remote`` connection to ``origin/master``. What remote repository is ``remote/origin``? Try ``git remote -v`` to see the URLs for the remote. They will point to your [github](https://github.com/numpy/numpy) fork.

    Now you want to connect to the upstream [NumPy github](https://github.com/numpy/numpy) repository, so you can merge in changes from trunk.

##### Linking your repository to the upstream repo

``` bash
cd numpy
git remote add upstream https://github.com/numpy/numpy.git
```

``upstream`` here is just the arbitrary name we’re using to refer to the main [NumPy](https://www.numpy.org/) repository at [NumPy github](https://github.com/numpy/numpy).

Just for your own satisfaction, show yourself that you now have a new ‘remote’, with ``git remote -v show``, giving you something like:

``` bash
upstream     https://github.com/numpy/numpy.git (fetch)
upstream     https://github.com/numpy/numpy.git (push)
origin       https://github.com/your-user-name/numpy.git (fetch)
origin       https://github.com/your-user-name/numpy.git (push)
```

To keep in sync with changes in NumPy, you want to set up your repository so it pulls from ``upstream`` by default. This can be done with:

``` bash
git config branch.master.remote upstream
git config branch.master.merge refs/heads/master
```

You may also want to have easy access to all pull requests sent to the NumPy repository:

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

## Git configuration

### Overview

Your personal [git](https://git-scm.com/) configurations are saved in the ``.gitconfig`` file in your home directory. Here is an example ``.gitconfig`` file:

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

You can edit this file directly or you can use the git config --global command:

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

To set up on another computer, you can copy your ``~/.gitconfig`` file, or run the commands above.

### In detail

#### user.name and user.email

It is good practice to tell [git](https://git-scm.com/) who you are, for labeling any changes you make to the code. The simplest way to do this is from the command line:

``` bash
git config --global user.name "Your Name"
git config --global user.email you@yourdomain.example.com
```

This will write the settings into your git configuration file, which should now contain a user section with your name and email:

```
[user]
      name = Your Name
      email = you@yourdomain.example.com
```

Of course you’ll need to replace ``Your Name`` and ``you@yourdomain.example.com`` with your actual name and email address.

#### Aliases

You might well benefit from some aliases to common commands.

For example, you might well want to be able to shorten ``git checkout`` to ``git co``. Or you may want to alias ``git diff --color-words`` (which gives a nicely formatted output of the diff) to ``git wdiff``

The following ``git config --global`` commands:

``` bash
git config --global alias.ci "commit -a"
git config --global alias.co checkout
git config --global alias.st "status -a"
git config --global alias.stat "status -a"
git config --global alias.br branch
git config --global alias.wdiff "diff --color-words"
```

will create an alias section in your .gitconfig file with contents like this:

```
[alias]
        ci = commit -a
        co = checkout
        st = status -a
        stat = status -a
        br = branch
        wdiff = diff --color-words
```

#### Editor

You may also want to make sure that your editor of choice is used

``` bash
git config --global core.editor vim
```

#### Merging

To enforce summaries when doing merges (~/.gitconfig file again):

``` bash
[merge]
   log = true
```

Or from the command line:

``` bash
git config --global merge.log true
```

## Two and three dots in difference specs

Thanks to Yarik Halchenko for this explanation.

Imagine a series of commits A, B, C, D… Imagine that there are two branches, topic and master. You branched topic off master when master was at commit ‘E’. The graph of the commits looks like this:

```
     A---B---C topic
     /
D---E---F---G master
```

Then:

``` bash
git diff master..topic
```

will output the difference from G to C (i.e. with effects of F and G), while:

``` bash
git diff master...topic
```

would output just differences in the topic branch (i.e. only A, B, and C).

## Additional Git Resources

### Tutorials and summaries

- [github help](https://help.github.com) has an excellent series of how-to guides.
- [learn.github](https://learn.github.com/) has an excellent series of tutorials
- The [pro git book](https://git-scm.com/book/) is a good in-depth book on git.
- A [git cheat sheet](http://cheat.errtheblog.com/s/git) is a page giving summaries of common commands.
- The [git user manual](https://www.kernel.org/pub/software/scm/git/docs/user-manual.html)
- The [git tutorial](https://www.kernel.org/pub/software/scm/git/docs/gittutorial.html)
- The [git community book](https://book.git-scm.com/)
- [git ready](http://www.gitready.com/) - a nice series of tutorials
- [git casts](http://www.gitcasts.com/) - video snippets giving git how-tos.
- [git magic](http://www-cs-students.stanford.edu/~blynn/gitmagic/index.html) - extended introduction with intermediate detail
- The [git parable](http://tom.preston-werner.com/2009/05/19/the-git-parable.html) is an easy read explaining the concepts behind git.
- Our own [git foundation](https://matthew-brett.github.com/pydagogue/foundation.html) expands on the [git parable](http://tom.preston-werner.com/2009/05/19/the-git-parable.html).
- Fernando Perez’ git page - [Fernando’s git page](http://www.fperez.org/py4science/git.html) - many links and tips
- A good but technical page on [git concepts](http://www.eecs.harvard.edu/~cduan/technical/git/)
- [git svn crash course](https://git-scm.com/course/svn.html): [git](https://git-scm.com/) for those of us used to [subversion](http://subversion.tigris.org/)

### Advanced git workflow

There are many ways of working with [git](https://git-scm.com/); here are some posts on the
rules of thumb that other projects have come up with:

- Linus Torvalds on [git management](http://kerneltrap.org/Linux/Git_Management)
- Linus Torvalds on [linux git workflow](https://www.mail-archive.com/dri-devel@lists.sourceforge.net/msg39091.html) .  Summary; use the git tools
to make the history of your edits as clean as possible; merge from
upstream edits as little as possible in branches where you are doing
active development.

### Manual pages online

You can get these on your own machine with (e.g) ``git help push`` or
(same thing) ``git push --help``, but, for convenience, here are the
online manual pages for some common commands:

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