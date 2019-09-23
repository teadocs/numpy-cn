# Releasing a Version

## How to Prepare a Release

This file gives an overview of what is necessary to build binary releases for
NumPy.

### Current build and release info

The current info on building and releasing NumPy and SciPy is scattered in
several places. It should be summarized in one place, updated, and where
necessary described in more detail. The sections below list all places where
useful info can be found.

#### Source tree

- INSTALL.rst.txt
- release.sh
- pavement.py

#### NumPy Docs

- [https://github.com/numpy/numpy/blob/master/doc/HOWTO_RELEASE.rst.txt](https://github.com/numpy/numpy/blob/master/doc/HOWTO_RELEASE.rst.txt)
- [http://projects.scipy.org/numpy/wiki/MicrosoftToolchainSupport](http://projects.scipy.org/numpy/wiki/MicrosoftToolchainSupport) (dead link)

#### SciPy.org wiki

- [https://www.scipy.org/Installing_SciPy](https://www.scipy.org/Installing_SciPy) and links on that page.
- [http://new.scipy.org/building/windows.html](http://new.scipy.org/building/windows.html) (dead link)

#### Doc wiki

- [http://docs.scipy.org/numpy/docs/numpy-docs/user/install.rst/](https://docs.scipy.org/numpy/docs/numpy-docs/user/install.rst/) (dead link)

#### Release Scripts

- [https://github.com/numpy/numpy-vendor](https://github.com/numpy/numpy-vendor)

### Supported platforms and versions

Python 2.7 and >=3.4 are the currently supported versions when building from
source.  We test NumPy against all these versions every time we merge code to
master.  Binary installers may be available for a subset of these versions (see
below).

#### OS X

Python 2.7 and >=3.4 are the versions for which we provide binary installers.
OS X versions >= 10.6 are supported.  We build binary wheels for OSX that are
compatible with Python.org Python, system Python, homebrew and macports - see
this [OSX wheel building summary](https://github.com/MacPython/wiki/wiki/Spinning-wheels) for details.

#### Windows

We build 32- and 64-bit wheels for Python 2.7, 3.4, 3.5 on Windows. Windows
XP, Vista, 7, 8 and 10 are supported.  We build NumPy using the MSVC compilers
on Appveyor, but we are hoping to update to a [mingw-w64 toolchain](https://mingwpy.github.io).  The Windows wheels use ATLAS for BLAS / LAPACK.

#### Linux

We build and ship [manylinux1](https://www.python.org/dev/peps/pep-0513)
wheels for NumPy.  Many Linux distributions include their own binary builds
of NumPy.

#### BSD / Solaris

No binaries are provided, but successful builds on Solaris and BSD have been
reported.

### Tool chain

We build all our wheels on cloud infrastructure - so this list of compilers is
for information and debugging builds locally.  See the ``.travis.yml`` and
``appveyor.yml`` scripts in the [numpy wheels](https://github.com/MacPython/numpy-wheels) repo for the definitive source
of the build recipes. Packages that are available using pip are noted.

#### Compilers

The same gcc version is used as the one with which Python itself is built on
each platform. At the moment this means:

- OS X builds on travis currently use  *clang* .  It appears that binary wheels
for OSX >= 10.6 can be safely built from the travis-ci OSX 10.9 VMs
when building against the Python from the Python.org installers;
- Windows builds use the MSVC version corresponding to the Python being built
against;
- Manylinux1 wheels use the gcc provided on the Manylinux docker images.

You will need Cython for building the binaries.  Cython compiles the ``.pyx``
files in the NumPy distribution to ``.c`` files.

#### Building source archives and wheels

You will need write permission for numpy-wheels in order to trigger wheel
builds.

- Python(s) from [python.org](https://python.org) or linux distro.
- cython
- virtualenv (pip)
- Paver (pip)
- numpy-wheels [https://github.com/MacPython/numpy-wheels](https://github.com/MacPython/numpy-wheels) (clone)

#### Building docs

Building the documents requires a number of latex ``.sty`` files. Install them
all to avoid aggravation.

- Sphinx (pip)
- numpydoc (pip)
- Matplotlib
- Texlive (or MikTeX on Windows)

#### Uploading to PyPI

- terryfy [https://github.com/MacPython/terryfy](https://github.com/MacPython/terryfy) (clone).
- beautifulsoup4 (pip)
- delocate (pip)
- auditwheel (pip)
- twine (pip)

#### Generating author/pr lists

You will need a personal access token
[https://help.github.com/article/creating-a-personal-access-token-for-the-command-line/](https://help.github.com/article/creating-a-personal-access-token-for-the-command-line/)
so that scripts can access the github NumPy repository.

- gitpython (pip)
- pygithub (pip)

#### Virtualenv

Virtualenv is a very useful tool to keep several versions of packages around.
It is also used in the Paver script to build the docs.

### What is released

#### Wheels

- Windows wheels for Python 2.7, 3.4, 3.5, for 32- and 64-bit, built using
Appveyor;
- Dual architecture OSX wheels built via travis-ci;
- 32- and 64-bit Manylinux1 wheels built via travis-ci.

See the [numpy wheels](https://github.com/MacPython/numpy-wheels) building repository for more detail.

#### Other

- Release Notes
- Changelog

#### Source distribution

We build source releases in both .zip and .tar.gz formats.

### Release process

#### Agree on a release schedule

A typical release schedule is one beta, two release candidates and a final
release.  It’s best to discuss the timing on the mailing list first, in order
for people to get their commits in on time, get doc wiki edits merged, etc.
After a date is set, create a new maintenance/x.y.z branch, add new empty
release notes for the next version in the master branch and update the Trac
Milestones.

#### Make sure current branch builds a package correctly

``` python
git clean -fxd
python setup.py bdist
python setup.py sdist
```

To actually build the binaries after everything is set up correctly, the
release.sh script can be used. For details of the build process itself, it is
best to read the pavement.py script.

::: tip Note

The following steps are repeated for the beta(s), release
candidates(s) and the final release.

:::

#### Check deprecations

Before the release branch is made, it should be checked that all deprecated
code that should be removed is actually removed, and all new deprecations say
in the docstring or deprecation warning at what version the code will be
removed.

#### Check the C API version number

The C API version needs to be tracked in three places

- numpy/core/setup_common.py
- numpy/core/code_generators/cversions.txt
- numpy/core/include/numpy/numpyconfig.h

There are three steps to the process.

1. If the API has changed, increment the C_API_VERSION in setup_common.py. The API is unchanged only if any code compiled against the current API will be backward compatible with the last released NumPy version. Any changes to C structures or additions to the public interface will make the new API not backward compatible.
1. If the C_API_VERSION in the first step has changed, or if the hash of the API has changed, the cversions.txt file needs to be updated. To check the hash, run the script numpy/core/cversions.py and note the API hash that is printed. If that hash does not match the last hash in numpy/core/code_generators/cversions.txt the hash has changed. Using both the appropriate C_API_VERSION and hash, add a new entry to cversions.txt. If the API version was not changed, but the hash differs, you will need to comment out the previous entry for that API version. For instance, in NumPy 1.9 annotations were added, which changed the hash, but the API was the same as in 1.8. The hash serves as a check for API changes, but it is not definitive.
1. If steps 1 and 2 are done correctly, compiling the release should not give a warning “API mismatch detect at the beginning of the build”.

The C ABI version number in numpy/core/setup_common.py should only be updated for a major release.

#### Check the release notes

Use [towncrier](https://github.com/hawkowl/towncrier) to build the release note, copy it to the proper name, and
commit the changes. This will remove all the fragments from ``changelog/*.rst``
and add ``doc/release/latest-note.rst`` which must be renamed with the proper
version number:

``` python
python -mtowncrier --version "Numpy 1.11.0"
git mv doc/release/latest-note.rst doc/release/1.11.0-notes.rst
git commit -m"Create release note"
```

Check that the release notes are up-to-date.

Update the release notes with a Highlights section. Mention some of the
following:

- major new features
- deprecated and removed features
- supported Python versions
- for SciPy, supported NumPy version(s)
- outlook for the near future

#### Update the release status and create a release “tag”

Identify the commit hash of the release, e.g. 1b2e1d63ff.

``` bash
git co 1b2e1d63ff # gives warning about detached head
```

First, change/check the following variables in pavement.py depending on the release version:

``` python
RELEASE_NOTES = 'doc/release/1.7.0-notes.rst'
LOG_START = 'v1.6.0'
LOG_END = 'maintenance/1.7.x'
```

Do any other changes. When you are ready to release, do the following changes:

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

And make sure the ``VERSION`` variable is set properly.

Now you can make the release commit and tag.  We recommend you don’t push
the commit or tag immediately, just in case you need to do more cleanup.  We
prefer to defer the push of the tag until we’re confident this is the exact
form of the released code (see: [Push the release tag and commit](#push-tag-and-commit)):

``` bash
git commit -s -m “REL: Release.” setup.py git tag -s <version>
```

The ``-s`` flag makes a PGP (usually GPG) signed tag.  Please do sign the
release tags.

The release tag should have the release number in the annotation (tag
message).  Unfortunately, the name of a tag can be changed without breaking the
signature, the contents of the message cannot.

See: [https://github.com/scipy/scipy/issues/4919](https://github.com/scipy/scipy/issues/4919) for a discussion of signing
release tags, and [https://keyring.debian.org/creating-key.html](https://keyring.debian.org/creating-key.html) for instructions
on creating a GPG key if you do not have one.

To make your key more readily identifiable as you, consider sending your key
to public keyservers, with a command such as:

``` python
gpg --send-keys <yourkeyid>
```

#### Apply patch to fix bogus strides

NPY_RELAXED_STRIDE_CHECKING was made the default in NumPy 1.10.0 and bogus
strides are used in the development branch to smoke out problems. The
[patch](https://github.com/numpy/numpy/pull/5996) should be updated if
necessary and applied to the release branch to rationalize the strides.

#### Update the version of the master branch

Increment the release number in setup.py. Release candidates should have “rc1”
(or “rc2”, “rcN”) appended to the X.Y.Z format.

Also create a new version hash in cversions.txt and a corresponding version
define NPY_x_y_API_VERSION in numpyconfig.h

#### Trigger the wheel builds on travis-ci and Appveyor

See the  *numpy wheels*  repository.

In that repository edit the files:

- ``.travis.yml``;
- ``appveyor.yml``.

In both cases, set the ``BUILD_COMMIT`` variable to the current release tag - e.g. ``v1.11.1``.

Make sure that the release tag has been pushed.

Trigger a build by doing a commit of your edits to ``.travis.yml`` and ``appveyor.yml`` to the repository:

``` python
cd /path/to/numpy-wheels
# Edit .travis.yml, appveyor.yml
git commit
git push
```

The wheels, once built, appear at a Rackspace container pointed at by:

- [http://wheels.scipy.org](http://wheels.scipy.org)
- [https://3f23b170c54c2533c070-1c8a9b3114517dc5fe17b7c3f8c63a43.ssl.cf2.rackcdn.com](https://3f23b170c54c2533c070-1c8a9b3114517dc5fe17b7c3f8c63a43.ssl.cf2.rackcdn.com)

The HTTP address may update first, and you should wait 15 minutes after the
build finishes before fetching the binaries.

#### Make the release

Build the changelog and notes for upload with:

``` python
paver write_release_and_log
```

#### Build and archive documentation

Do:

``` python
cd doc/
make dist
```

to check that the documentation is in a buildable state. Then, after tagging,
create an archive of the documentation in the numpy/doc repo:

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

#### Update PyPI

The wheels and source should be uploaded to PyPI.

You should upload the wheels first, and the source formats last, to make sure
that pip users don’t accidentally get a source install when they were
expecting a binary wheel.

You can do this automatically using the ``wheel-uploader`` script from
[https://github.com/MacPython/terryfy](https://github.com/MacPython/terryfy).  Here is the recommended incantation for
downloading all the Windows, Manylinux, OSX wheels and uploading to PyPI.

``` python
NPY_WHLS=~/wheelhouse   # local directory to cache wheel downloads
CDN_URL=https://3f23b170c54c2533c070-1c8a9b3114517dc5fe17b7c3f8c63a43.ssl.cf2.rackcdn.com
wheel-uploader -u $CDN_URL -w $NPY_WHLS -v -s -t win numpy 1.11.1rc1
wheel-uploader -u $CDN_URL -w warehouse -v -s -t macosx numpy 1.11.1rc1
wheel-uploader -u $CDN_URL -w warehouse -v -s -t manylinux1 numpy 1.11.1rc1
```

The ``-v`` flag gives verbose feedback, ``-s`` causes the script to sign the
wheels with your GPG key before upload. Don’t forget to upload the wheels
before the source tarball, so there is no period for which people switch from
an expected binary install to a source install from PyPI.

There are two ways to update the source release on PyPI, the first one is:

``` python
$ git clean -fxd  # to be safe
$ python setup.py sdist --formats=gztar,zip  # to check
# python setup.py sdist --formats=gztar,zip upload --sign
```

This will ask for your key PGP passphrase, in order to sign the built source
packages.

The second way is to upload the PKG_INFO file inside the sdist dir in the
web interface of PyPI. The source tarball can also be uploaded through this
interface.

#### Push the release tag and commit

Finally, now you are confident this tag correctly defines the source code that
you released you can push the tag and release commit up to github:

``` python
git push  # Push release commit
git push upstream <version>  # Push tag named <version>
```

where ``upstream`` points to the main [https://github.com/numpy/numpy.git](https://github.com/numpy/numpy.git)
repository.

#### Update scipy.org

A release announcement with a link to the download site should be placed in the
sidebar of the front page of scipy.org.

The scipy.org should be a PR at [https://github.com/scipy/scipy.org](https://github.com/scipy/scipy.org). The file
that needs modification is ``www/index.rst``. Search for ``News``.

#### Announce to the lists

The release should be announced on the mailing lists of
NumPy and SciPy, to python-announce, and possibly also those of
Matplotlib, IPython and/or Pygame.

During the beta/RC phase, an explicit request for testing the binaries with
several other libraries (SciPy/Matplotlib/Pygame) should be posted on the
mailing list.

#### Announce to Linux Weekly News

Email the editor of LWN to let them know of the release.  Directions at:
[https://lwn.net/op/FAQ.lwn#contact](https://lwn.net/op/FAQ.lwn#contact)

#### After the final release

After the final release is announced, a few administrative tasks are left to be
done:

- Forward port changes in the release branch to release notes and release
scripts, if any, to master branch.
- Update the Milestones in Trac.

## Step-by-Step Directions

This file contains a walkthrough of the NumPy 1.14.5 release on Linux.
The commands can be copied into the command line, but be sure to
replace 1.14.5 by the correct version.

### Release  Walkthrough

Note that in the code snippets below, ``upstream`` refers to the root repository on
github and ``origin`` to a fork in your personal account. You may need to make adjustments
if you have not forked the repository but simply cloned it locally. You can
also edit ``.git/config`` and add ``upstream`` if it isn’t already present.

#### Backport Pull Requests

Changes that have been marked for this release must be backported to the
maintenance/1.14.x branch.

#### Update Release documentation

The file ``doc/changelog/1.14.5-changelog.rst`` should be updated to reflect
the final list of changes and contributors. This text can be generated by:

``` python
$ python tools/changelog.py $GITHUB v1.14.4..maintenance/1.14.x > doc/changelog/1.14.5-changelog.rst
```

where ``GITHUB`` contains your github access token. This text may also be
appended to ``doc/release/1.14.5-notes.rst`` for release updates, though not
for new releases like ``1.14.0``, as the changelogs for ``*.0`` releases tend to be
excessively long. The ``doc/source/release.rst`` file should also be
updated with a link to the new release notes. These changes should be committed
to the maintenance branch, and later will be forward ported to master.

#### Finish the Release Note

Fill out the release note ``doc/release/1.14.5-notes.rst`` calling out
significant changes.

#### Prepare the release commit

Checkout the branch for the release, make sure it is up to date, and clean the
repository:

``` python
$ git checkout maintenance/1.14.x
$ git pull upstream maintenance/1.14.x
$ git submodule update
$ git clean -xdf > /dev/null
```

Edit pavement.py and setup.py as detailed in HOWTO_RELEASE:

``` python
$ gvim pavement.py setup.py
$ git commit -a -m"REL: NumPy 1.14.5 release."
```

Sanity check:

``` python
$ python runtests.py -m "full"  # NumPy < 1.17 only
$ python3 runtests.py -m "full"
```

Push this release directly onto the end of the maintenance branch. This
requires write permission to the numpy repository:

``` python
$ git push upstream maintenance/1.14.x
```

As an example, see the 1.14.3 REL commit: [https://github.com/numpy/numpy/commit/73299826729be58cec179b52c656adfcaefada93](https://github.com/numpy/numpy/commit/73299826729be58cec179b52c656adfcaefada93).

#### Build source releases

Paver is used to build the source releases. It will create the ``release`` and
``release/installers`` directories and put the ``*.zip`` and ``*.tar.gz``
source releases in the latter.

``` python
$ cython --version  # check that you have the correct cython version
$ paver sdist  # sdist will do a git clean -xdf, so we omit that
```

#### Build wheels

Trigger the wheels build by pointing the numpy-wheels repository at this
commit. This can take a while. The numpy-wheels repository is cloned from
[https://github.com/MacPython/numpy-wheels](https://github.com/MacPython/numpy-wheels). Start with a pull as the repo
may have been accessed and changed by someone else and a push will fail:

``` python
$ cd ../numpy-wheels
$ git pull upstream master
$ git branch <new version>  # only when starting new numpy version
$ git checkout v1.14.x  # v1.14.x already existed for the 1.14.4 release
```

Edit the ``.travis.yml`` and ``.appveyor.yml`` files to make sure they have the
correct version, and put in the commit hash for the ``REL`` commit created
above for ``BUILD_COMMIT``, see the _example from  *v1.14.3* :

``` python
$ gvim .travis.yml .appveyor.yml
$ git commit -a
$ git push upstream HEAD
```

Now wait. If you get nervous at the amount of time taken – the builds can take
several hours– you can check the build progress by following the links
provided at [https://github.com/MacPython/numpy-wheels](https://github.com/MacPython/numpy-wheels) to check the travis
and appveyor build status. Check if all the needed wheels have been built and
uploaded before proceeding. There should currently be 22 of them at
[https://wheels.scipy.org](https://wheels.scipy.org), 4 for Mac, 8 for Windows, and 10 for Linux.
Note that sometimes builds, like tests, fail for unrelated reasons and you will
need to restart them.

#### Download wheels

When the wheels have all been successfully built, download them using the ``wheel-uploader``
in the ``terryfy`` repository.  The terryfy repository may be cloned from
[https://github.com/MacPython/terryfy](https://github.com/MacPython/terryfy) if you don’t already have it.  The
wheels can also be uploaded using the ``wheel-uploader``, but we prefer to
download all the wheels to the ``../numpy/release/installers`` directory and
upload later using ``twine``:

``` python
$ cd ../terryfy
$ git pull upstream master
$ CDN_URL=https://3f23b170c54c2533c070-1c8a9b3114517dc5fe17b7c3f8c63a43.ssl.cf2.rackcdn.com
$ NPY_WHLS=../numpy/release/installers
$ ./wheel-uploader -u $CDN_URL -n -v -w $NPY_WHLS -t win numpy 1.14.5
$ ./wheel-uploader -u $CDN_URL -n -v -w $NPY_WHLS -t manylinux1 numpy 1.14.5
$ ./wheel-uploader -u $CDN_URL -n -v -w $NPY_WHLS -t macosx numpy 1.14.5
```

If you do this often, consider making CDN_URL and NPY_WHLS part of your default
environment.

#### Generate the README files

This needs to be done after all installers are downloaded, but before the pavement
file is updated for continued development:

``` python
$ cd ../numpy
$ paver write_release
```

#### Tag the release

Once the wheels have been built and downloaded without errors, go back to your
numpy repository in the maintenance branch and tag the ``REL`` commit, signing
it with your gpg key:

``` python
$ git tag -s v1.14.5
```

You should upload your public gpg key to github, so that the tag will appear
“verified” there.

Check that the files in ``release/installers`` have the correct versions, then
push the tag upstream:

``` python
$ git push upstream v1.14.5
```

We wait until this point to push the tag because it is public and should not
be changed after it has been pushed.

#### Reset the maintenance branch into a development state

Add another ``REL`` commit to the numpy maintenance branch, which resets the
``ISREALEASED`` flag to ``False`` and increments the version counter:

``` python
$ gvim pavement.py setup.py
```

Create release notes for next release and edit them to set the version:

``` python
$ cp doc/release/template.rst doc/release/1.14.6-notes.rst
$ gvim doc/release/1.14.6-notes.rst
$ git add doc/release/1.14.6-notes.rst
```

Add new release notes to the documentation release list:

``` python
$ gvim doc/source/release.rst
```

Commit the result:

``` python
$ git commit -a -m"REL: prepare 1.14.x for further development"
$ git push upstream maintenance/1.14.x
```

#### Upload to PyPI

Upload to PyPI using ``twine``. A recent version of ``twine`` of is needed
after recent PyPI changes, version ``1.11.0`` was used here.

``` sh
$ cd ../numpy
$ twine upload release/installers/*.whl
$ twine upload release/installers/numpy-1.14.5.zip  # Upload last.
```

If one of the commands breaks in the middle, which is not uncommon, you may
need to selectively upload the remaining files because PyPI does not allow the
same file to be uploaded twice. The source file should be uploaded last to
avoid synchronization problems if pip users access the files while this is in
process. Note that PyPI only allows a single source distribution, here we have
chosen the zip archive.

#### Upload files to github

Go to [https://github.com/numpy/numpy/releases](https://github.com/numpy/numpy/releases), there should be a ``v1.14.5
tag``, click on it and hit the edit button for that tag. There are two ways to
add files, using an editable text window and as binary uploads.

- Cut and paste the ``release/README.md`` file contents into the text window.
- Upload ``release/installers/numpy-1.14.5.tar.gz`` as a binary file.
- Upload ``release/installers/numpy-1.14.5.zip`` as a binary file.
- Upload ``release/README.rst`` as a binary file.
- Upload ``doc/changelog/1.14.5-changelog.rst`` as a binary file.
- Check the pre-release button if this is a pre-releases.
- Hit the ``{Publish,Update} release`` button at the bottom.

#### Upload documents to docs.scipy.org

This step is only needed for final releases and can be skipped for
pre-releases. You will also need upload permission for the document server, if
you do not have permission ping Pauli Virtanen or Ralf Gommers to generate and
upload the documentation. Otherwise:

``` python
$ pushd doc
$ make dist
$ make upload USERNAME=<yourname> RELEASE=v1.14.5
$ popd
```

If the release series is a new one, you will need to rebuild and upload the
``docs.scipy.org`` front page:

``` python
$ cd ../docs.scipy.org
$ gvim index.rst
```

Note: there is discussion about moving the docs to github. This section will be
updated when/if that happens.

#### Announce the release on scipy.org

This assumes that you have forked [https://github.com/scipy/scipy.org](https://github.com/scipy/scipy.org):

``` python
$ cd ../scipy.org
$ git checkout master
$ git pull upstream master
$ git checkout -b numpy-1.14.5
$ gvim www/index.rst # edit the News section
$ git commit -a
$ git push origin HEAD
```

Now go to your fork and make a pull request for the branch.

#### Announce to mailing lists

The release should be announced on the numpy-discussion, scipy-devel,
scipy-user, and python-announce-list mailing lists. Look at previous
announcements for the basic template. The contributor and PR lists are the same
as generated for the release notes above. If you crosspost, make sure that
python-announce-list is BCC so that replies will not be sent to that list.

#### Post-Release Tasks

Forward port the documentation changes ``doc/release/1.14.5-notes.rst``,
``doc/changelog/1.14.5-changelog.rst`` and add the release note to
``doc/source/release.rst``.
