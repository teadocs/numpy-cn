# Building the NumPy API and reference docs

We currently use [Sphinx](https://www.sphinx-doc.org/) for generating the API and reference
documentation for NumPy.  You will need Sphinx 1.8.3 or newer.

If you only want to get the documentation, note that pre-built
versions can be found at

  - [https://docs.scipy.org/](https://docs.scipy.org/)

in several different formats.

## Instructions

If you obtained NumPy via git, get also the git submodules that contain
additional parts required for building the documentation:

``` python
git submodule init
git submodule update
```

In addition, building the documentation requires the Sphinx extension
 *plot_directive* , which is shipped with [Matplotlib](https://matplotlib.org/). This Sphinx extension can
be installed by installing Matplotlib. You will also need python3.6.

Since large parts of the main documentation are obtained from numpy via
``import numpy`` and examining the docstrings, you will need to first build
NumPy, and install it so that the correct version is imported.

Note that you can eg. install NumPy to a temporary location and set
the PYTHONPATH environment variable appropriately.

After NumPy is installed, install SciPy since some of the plots in the random
module require [``scipy.special``](https://docs.scipy.org/doc/scipy/reference/special.html#module-scipy.special) to display properly. Now you are ready to
generate the docs, so write:

``` python
make html
```

in the ``doc/`` directory. If all goes well, this will generate a
``build/html`` subdirectory containing the built documentation. If you get
a message about ``installed numpy != current repo git version``, you must
either override the check by setting ``GITVER`` or re-install NumPy.

Note that building the documentation on Windows is currently not actively
supported, though it should be possible. (See [Sphinx](https://www.sphinx-doc.org/) documentation
for more information.)

To build the PDF documentation, do instead:

``` python
make latex
make -C build/latex all-pdf
```

You will need to have Latex installed for this.

Instead of the above, you can also do:

``` python
make dist
```

which will rebuild NumPy, install it to a temporary location, and
build the documentation in all formats. This will most likely again
only work on Unix platforms.

The documentation for NumPy distributed at [https://docs.scipy.org](https://docs.scipy.org) in html and
pdf format is also built with ``make dist``.  See [HOWTO RELEASE](https://github.com/numpy/numpy/blob/master/doc/HOWTO_RELEASE.rst.txt) for details on
how to update [https://docs.scipy.org](https://docs.scipy.org).

## Sphinx extensions

NumPy’s documentation uses several custom extensions to Sphinx.  These
are shipped in the ``sphinxext/`` directory (as git submodules, as discussed
above), and are automatically enabled when building NumPy’s documentation.

If you want to make use of these extensions in third-party
projects, they are available on [PyPi](https://pypi.org/) as the [numpydoc](https://python.org/pypi/numpydoc) package.