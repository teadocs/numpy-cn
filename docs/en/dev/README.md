# Contributing to NumPy

Not a coder? Not a problem! NumPy is multi-faceted, and we can use a lot of help.
These are all activities we’d like to get help with (they’re all important, so
we list them in alphabetical order):

- Code maintenance and development
- Community coordination
- DevOps
- Developing educational content & narrative documentation
- Writing technical documentation
- Fundraising
- Project management
- Marketing
- Translating content
- Website design and development

The rest of this document discusses working on the NumPy code base and documentation.
We’re in the process of updating our descriptions of other activities and roles.
If you are interested in these other activities, please contact us!
You can do this via
the [numpy-discussion mailing list](https://scipy.org/scipylib/mailing-lists.html),
or on GitHub (open an issue or comment on a relevant issue). These are our preferred
communication channels (open source is open by nature!), however if you prefer
to discuss in private first, please reach out to our community coordinators
at  *numpy-team@googlegroups.com*  or  *numpy-team.slack.com*  (send an email to
 *numpy-team@googlegroups.com*  for an invite the first time).

## Development process - summary

Here’s the short summary, complete TOC links are below:

1. If you are a first-time contributor:

  - Go to https://github.com/numpy/numpy and click the “fork” button to create your own copy of the project.

  - Clone the project to your local computer:

  ``` bash
  git clone https://github.com/your-username/numpy.git
  ```

  - Change the directory:

    ``` bash
    cd numpy
    ```

  - Add the upstream repository:

    ``` bash
    git remote add upstream https://github.com/numpy/numpy.git
    ```

  - Now, git remote -v will show two remote repositories named:
    - upstream, which refers to the numpy repository
    - origin, which refers to your personal fork
1. Develop your contribution:

    - Pull the latest changes from upstream:

    ``` bash
    git checkout master
    git pull upstream master
    ```

    - Create a branch for the feature you want to work on. Since the branch name will appear in the merge message, use a sensible name such as ‘linspace-speedups’:

    ``` bash
    git checkout -b linspace-speedups
    ```

    - Commit locally as you progress (git add and git commit) Use a properly formatted commit message, write tests that fail before your change and pass afterward, run all the tests locally. Be sure to document any changed behavior in docstrings, keeping to the NumPy docstring standard.
1. To submit your contribution:
    - Push your changes back to your fork on GitHub:

    ``` bash
    git push origin linspace-speedups
    ```

    - Enter your GitHub username and password (repeat contributors or advanced users can remove this step by connecting to GitHub with SSH .

    - Go to GitHub. The new branch will show up with a green Pull Request button. Make sure the title and message are clear, concise, and self- explanatory. Then click the button to submit it.

    - If your commit introduces a new feature or changes functionality, post on the mailing list to explain your changes. For bug fixes, documentation updates, etc., this is generally not necessary, though if you do not get any reaction, do feel free to ask for review.
1. Review process:
    - Reviewers (the other developers and interested community members) will write inline and/or general comments on your Pull Request (PR) to help you improve its implementation, documentation and style. Every single developer working on the project has their code reviewed, and we’ve come to see it as friendly conversation from which we all learn and the overall code quality benefits. Therefore, please don’t let the review discourage you from contributing: its only aim is to improve the quality of project, not to criticize (we are, after all, very grateful for the time you’re donating!).
    - To update your PR, make your changes on your local repository, commit, run tests, and only if they succeed push to your fork. As soon as those changes are pushed up (to the same branch as before) the PR will update automatically. If you have no idea how to fix the test failures, you may push your changes anyway and ask for help in a PR comment.
    - Various continuous integration (CI) services are triggered after each PR update to build the code, run unit tests, measure code coverage and check coding style of your branch. The CI tests must pass before your PR can be merged. If CI fails, you can find out why by clicking on the “failed” icon (red cross) and inspecting the build and test log. To avoid overuse and waste of this resource, test your work locally before committing.
    - A PR must be approved by at least one core team member before merging. Approval means the core team member has carefully reviewed the changes, and the PR is ready for merging.
1. Document changes

    Beyond changes to a functions docstring and possible description in the general documentation, if your change introduces any user-facing modifications, update the current release notes under doc/release/X.XX-notes.rst

    If your change introduces a deprecation, make sure to discuss this first on GitHub or the mailing list first. If agreement on the deprecation is reached, follow NEP 23 deprecation policy to add the deprecation.
1. Cross referencing issues

    If the PR relates to any issues, you can add the text xref gh-xxxx where xxxx is the number of the issue to github comments. Likewise, if the PR solves an issue, replace the xref with closes, fixes or any of the other flavors github accepts.

    In the source code, be sure to preface any issue or PR reference with gh-xxxx.

For a more detailed discussion, read on and follow the links at the bottom of this page.

### Divergence between ``upstream/master`` and your feature branch

If GitHub indicates that the branch of your Pull Request can no longer
be merged automatically, you have to incorporate changes that have been made
since you started into your branch. Our recommended way to do this is to
[rebase on master](development_workflow.html#rebasing-on-master).

### Guidelines

- All code should have tests (see [test coverage](#test-coverage) below for more details).
- All code should be [documented](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard).
- No changes are ever committed without review and approval by a core
team member.Please ask politely on the PR or on the [mailing list](https://mail.python.org/mailman/listinfo/numpy-devel) if you
get no response to your pull request within a week.

### Stylistic Guidelines

- Set up your editor to follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) (remove trailing white space, no tabs, etc.).  Check code with
pyflakes / flake8.
- Use numpy data types instead of strings (``np.uint8`` instead of
``"uint8"``).
- Use the following import conventions:

    ``` python
    import numpy as np
    ```

- For C code, see the [numpy-c-style-guide](style_guide.html#style-guide)

### Test coverage

Pull requests (PRs) that modify code should either have new tests, or modify existing
tests to fail before the PR and pass afterwards. You should [run the tests](development_environment.html#development-environment) before pushing a PR.

Tests for a module should ideally cover all code in that module,
i.e., statement coverage should be at 100%.

To measure the test coverage, install
[pytest-cov](https://pytest-cov.readthedocs.io/en/latest/)
and then run:

``` python
$ python runtests.py --coverage
```

This will create a report in ``build/coverage``, which can be viewed with:

``` python
$ firefox build/coverage/index.html
```

### Building docs

To build docs, run ``make`` from the ``doc`` directory. ``make help`` lists
all targets. For example, to build the HTML documentation, you can run:

``` python
make html
```

Then, all the HTML files will be generated in ``doc/build/html/``.
Since the documentation is based on docstrings, the appropriate version of
numpy must be installed in the host python used to run sphinx.

#### Requirements

[Sphinx](https://www.sphinx-doc.org/en/stable/) is needed to build
the documentation. Matplotlib and SciPy are also required.

#### Fixing Warnings

- “citation not found: R###” There is probably an underscore after a
reference in the first line of a docstring (e.g. [1]_). Use this
method to find the source file: $ cd doc/build; grep -rin R####
- “Duplicate citation R###, other instance in…”” There is probably a
[2] without a [1] in one of the docstrings

## Development process - details

The rest of the story

- [NumPy Code of Conduct](conduct/code_of_conduct.html)
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

NumPy-specific workflow is in [numpy-development-workflow](development_workflow.html#development-workflow).