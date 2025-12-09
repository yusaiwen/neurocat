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

Overview
--------

Neurocat is a lightweight Python package for neuroimaging, with a focus on surface-based data. It offers utilities for visualization, color map creation and manipulation, format conversion (e.g., CIFTI and GIFTI), surface I/O, and other surface-specific data operations. The project was created by an ailurophile (cat-lover) and aims to be simple, flexible, and well-documented.

Key features
------------

- Surface visualization and plotting utilities (neurocat.plotting)
- Colormap and gradient utilities (neurocat.color)
- Data transformations and medial-wall handling for surface meshes (neurocat.transfer)
- Utility functions for brain models and density estimation (neurocat.util)
- File I/O for common neuroimaging surface formats (neurocat.io)

Requirements
------------

- Python 3.10 or newer (development and testing conducted on 3.10)
- Connectome Workbench: required for ``neurocat.transfer.f2f`` function if you need. Remember to add ``wb_command`` to PATH environment.

Linux-specific notes
--------------------

On Linux, you may need to install either vtk-egl (for systems with GPU support) or vtk-osmesa (for headless/CPU-only environments). These wheels are served from the VTK index rather than PyPI:

.. code-block:: bash

    # For systems with GPU-enabled headless rendering
    pip install --extra-index-url https://wheels.vtk.org vtk-egl

    # For headless / CPU-only rendering
    pip install --extra-index-url https://wheels.vtk.org vtk-osmesa

Installation
------------

Install via pip from GitHub:

.. code-block:: bash

    pip install git+https://github.com/yusaiwen/neurocat.git

Quick start
-----------

Import the package in your Python project or interactive session to get started:

.. code-block:: python

    import neurocat
    # See the documentation for examples and module-level usage.

For detailed usage, examples, and API documentation, visit the project documentation at:
https://neurocat.readthedocs.io/en/latest/

Contributing
------------

Contributions are welcome! For development guidelines and contribution processes, please open issues or pull requests on GitHub and refer to the CONTRIBUTING file if present.

License
-------

This project is licensed under the MIT License — see the LICENSE file for details.

