neurocat
========

.. image:: https://img.shields.io/pypi/v/neurocat.svg
    :target: https://pypi.org/project/neurocat/
    :alt: PyPI version

.. image:: https://img.shields.io/badge/docs-latest-brightgreen.svg
    :target: https://neurocat.readthedocs.io/en/latest/
    :alt: Documentation Status

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :target: https://opensource.org/license/mit
    :alt: License

Description
-----------

Neurocat is a Python package for neuroimaging data(especially **surface data**) processing by an ailurophile🐱. It provides utilities for surface visualization, color manipulation, data transfer between formats (e.g., CIFTI and GIFTI), I/O operations, and surface data handling. Key features include:

- Surface data visualization (via ``neurocat.plotting``)
- Color gradients and colormaps (via ``neurocat.color``)
- Data transformation and medial wall removal (via ``neurocat.transfer``)
- Utility functions for brain models and densities (via ``neurocat.util``)
- Saving and loading neuroimaging files (via ``neurocat.io``)

Installation requirements
-------------------------
Currently, ``neurocat`` works with Python 3.10+(the developer only test on 3.10). 

You will also need to have Connectome Workbench installed and available on your path in order to use most of the transformation / resampling functionality of neuromaps.

Installation
------------

If you are using Linux, install vtk-egl first:

.. code-block:: bash

    pip install --extra-index-url https://wheels.vtk.org vtk-egl

Then install neurocat:

.. code-block:: bash

    pip install git+https://github.com/yusaiwen/neurocat.git

Usage
-----

Import and use modules:

.. code-block:: python

    from neurocat.color import get_color_gradient
    from neurocat.transfer import remove_mw
    from neurocat.util import judge_density
    from neurocat.io import save_cii

    # Example: Create a color gradient
    gradient = get_color_gradient('#FF0000', '#0000FF', 10)

    # Example: Remove medial wall from data
    cleaned_data = remove_mw(data)

Contributing
------------

Contributions are welcome! Please submit issues or pull requests on GitHub.

License
-------

This project is licensed under the MIT License - see the LICENSE file for details.

