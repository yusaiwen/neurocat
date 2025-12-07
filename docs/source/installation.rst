Installation and Setup
======================


Requirements
------------

The ``f2f`` function in the ``transfer`` module relies on `Connectome Workbench <https://www.humanconnectome.org/software/connectome-workbench>`_.
If you want to use this function, install Connectome Workbench and ensure ``wb_command`` is available in your system's PATH.

On Linux and macOS, add the following line to your ``.bashrc`` file (assuming you are using Bash):

.. code-block:: bash
    
    export PATH=/PATH/TO/WORKBENCH/bin_linux64:$PATH

On Windows:

.. code-block:: powershell

    setx PATH "%PATH%;C:\PATH\TO\WORKBENCH\bin_windows64"

After installation, open your terminal and run:

.. code-block:: bash

    wb_command -version

to verify it is installed correctly.


Basic Installation
------------------

Neurocat supports Python 3.10+. The developer has tested it primarily on Python 3.10 for stability.
Install ``neurocat`` from the source repository using:

.. code-block:: bash

    pip install git+https://github.com/yusaiwen/neurocat.git

Alternatively, clone the repository and install from the local directory:

.. code-block:: bash

    git clone https://github.com/yusaiwen/neurocat.git
    cd neurocat
    pip install .
