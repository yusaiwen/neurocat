# neurocat

[![hatch-managed](https://img.shields.io/badge/hatch-managed-blue)](https://hatch.pypa.io/latest/) ![License](https://img.shields.io/badge/license-MIT-yellow) ![Language](https://img.shields.io/badge/language-python-brightgreen) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/neurocat.svg)](https://pypi.org/project/neurocat)

Neurocat is an fmri surface processing and visualization software by ailurophil😽.

-----

## Table of Contents

- [Installation](#installation)
- [License](#license)

## Installation

Install vtk-egl if you use Linux server.

```
pip install --extra-index-url https://wheels.vtk.org vtk-egl
```

Otherwise install vtk normal version.

```
pip install vtk
```

And you can intall neurocat:

```console
pip install neurocat
```

## Windows users

For window users, you should install the following softwares mannually:

Many operation was done by wand, whoes underline dependency is imagemagick, you should install it mannually:

https://docs.wand-py.org/en/latest/guide/install.html#install-imagemagick-on-windows

Imagemagick uses ghostcript to read pdf file.

https://ghostscript.com/releases/gsdnld.html



Remember that Windows may give many unexpected erros, use Linux or Macos instead if the bugs cannot be fixed.

## License

`neurocat` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
