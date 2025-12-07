.. _usage_plotting:

Plotting Brain Surfaces
=======================

Overview
--------

The ``neurocat.plotting`` module provides tools to visualize cortical and subcortical brain data
on surface meshes. The primary interface is 
* ``BrainPlotter`` class, which offers:

  
  * ``draw_surface`` – cortical data (fsLR surfaces, Schaefer/Glasser atlases)
  * ``draw_subcortex`` – subcortical data (Tian 2020 atlas)
  * ``draw_tian2020`` – combined cortical + subcortical visualization

All methods support both vertex-wise and atlas-based input, with automatic outline detection and output to PNG/PDF formats.


BrainPlotter
------------



- Maintains plotting configuration (cmap, color_range, layout, fig_name, tp, mesh, etc.).
- Keyword arguments override values in ``PlotConfig``.
- Produces PNG/PDF outputs and can display inline (when ``show=True``).

Example:

.. code-block:: python

    import numpy as np
    from neurocat.plotting import BrainPlotter, PlotConfig

    # Default plotter
    plotter = BrainPlotter()

    # Custom config (overridable via method kwargs)
    cfg = PlotConfig(cmap='viridis', layout='grid', fig_name='demo_brain')
    plotter = BrainPlotter(cfg)


Methods
-------

draw_surface
~~~~~~~~~~~~

Render cortical data on fsLR surfaces. Accepts vertex-wise data (59412/64984) or atlas data
(e.g., Schaefer, Glasser). Atlas data are converted to whole-brain and outlined by default.

Signature: ``draw_surface(data, **kwargs) -> surfplot.Plot``

Key kwargs: ``cmap``, ``color_range``, ``layout``, ``fig_name``, ``tp``, ``mesh``,
``trim``, ``pn``, ``show``, ``cbar_label``, ``system``, ``sulc``, ``outline``, ``force_nooutline``.

Example:

.. code-block:: python

    import numpy as np
    from neurocat.plotting import BrainPlotter

    # Schaefer-400 example data
    data = np.random.rand(400)
    plotter = BrainPlotter()
    plotter.draw_surface(
        data,
        fig_name='cortex_schaefer400',
        cmap='cividis',
        layout='grid',
        sulc=True,
        legend='Z-score'
    )
    # Outputs cortex_schaefer400.png and cortex_schaefer400.pdf

draw_subcortex
~~~~~~~~~~~~~~

Render subcortical data using Tian 2020 atlas (32 regions total, split L/R).

Signature: ``draw_subcortex(data, **kwargs) -> None``

Key kwargs: ``cmap``, ``color_range``, ``fig_name``, ``trim``.

Example:

.. code-block:: python

    import numpy as np
    from neurocat.plotting import BrainPlotter

    # Tian S4 subcortex data (32 values)
    sub_data = np.random.rand(32)
    plotter = BrainPlotter()
    plotter.draw_subcortex(
        sub_data,
        fig_name='sub_tian',
        cmap='plasma',
        trim=True  # also saves *_lh_trim.png, *_rh_trim.png
    )
    # Outputs sub_tian_lh.png, sub_tian_rh.png, and trimmed variants

draw_tian2020
~~~~~~~~~~~~~

Create a combined plot: cortical (Schaefer/Glasser) + subcortical (Tian 2020).
Input length must be 432 (Schaefer400 + Tian32) or 1032.

Signature: ``draw_tian2020(data, **kwargs) -> surfplot.Plot``

Key kwargs: same as ``draw_surface`` plus ``just_mesh`` (for mesh-only).

Example:

.. code-block:: python

    import numpy as np
    from neurocat.plotting import BrainPlotter

    # Combined data: Schaefer-400 (400) + Tian (32) = 432
    combined = np.random.rand(432)
    plotter = BrainPlotter()
    plotter.draw_tian2020(
        combined,
        fig_name='combined_brain',
        cmap='viridis',
        layout='grid',
        legend='Intensity'
    )
    # Outputs combined_brain.png

Legacy Functions
----------------

For backward compatibility:
- ``draw_and_save_hm``: single hemisphere heatmap (deprecated).
- ``draw_and_save``: calls ``BrainPlotter.draw_surface``.
- ``draw_subcortex_tian``: calls ``BrainPlotter.draw_subcortex``.
- ``draw_sftian_save``: calls ``BrainPlotter.draw_tian2020``.

Prefer using ``BrainPlotter`` directly in new code.

Mesh Utility
------------

Plot only the brain mesh (no data):

.. code-block:: python

    from neurocat.plotting import plot_mesh
    plot_mesh(tp='fslr', mesh='veryinflated', hm='L', fig_name='mesh_left')
    # Outputs mesh_left.png







