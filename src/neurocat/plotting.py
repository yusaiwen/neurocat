"""
Brain plotting utilities for neuroimaging data.
This module provides tools for visualizing brain data on cortical surfaces and subcortical regions using various atlases and templates. It supports plotting heatmaps, outlines, and combined surface-subcortical views with customizable configurations.
"""

# import the basic package
import shutil
import numpy as np
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Optional, Union, List, Tuple

from deprecated.sphinx import deprecated

from wand.image import Image as wand_img
from wand.color import Color

import nibabel as nib

from surfplot import Plot  # for plot surface
from templateflow.api import get as tpget

import pyvista as pv  # pyvista for plot subcortex

# cope with image
from reportlab.pdfgen import canvas
from PIL import Image
from IPython.display import Image as IPyImage, display
from .color import *
from .util import (FSLR,
                   con_path_list,
                   __base__,
                   tmp_name,
                   cleanup_files,
                   display_pdf2,
                   second_smallest,
                   get_nomw_vertex_n)
from .transfer import (
    reverse_mw,
    atlas_2_wholebrain as a2w,
    tian_a2w as ta2w
)

nib.imageglobals.logger.setLevel(40)
pv.set_jupyter_backend('static')  # to avoid opening a new window


# Constants
_LAYOUT_CONFIG = {
    'grid': {'size': (1200, 850), 'zoom': 1.7},
    'column': {'size': (600, 1600), 'zoom': 1.7},
    'row': {'size': (1600, 400), 'zoom': 1.25},
    'row_hm': {'size': (800, 400), 'zoom': 1.25},
    'column_hm': {'size': (600, 800), 'zoom': 1.7}
}
_FSLR_LEN = (59412,  # fslr 32k without medial wall
             64984   # fslr 32k with medial wall
       )
_ATLAS_LEN = tuple(range(100,1001,100)) + (  # Schaefer
            998,  # Schaefer 100 fslr lack two labels
            360,  # Glasser
            )


def _get_mesh_paths(tp, mesh) -> list:
    """
    Get the file paths for the specified template and mesh type.

    Parameters
    ----------
    tp : str
        Template type ('s1200' or 'fslr').
    mesh : str
        Mesh inflation type ('inflated' or 'veryinflated').

    Returns
    -------
    list
        List of file paths for the mesh surfaces.
    """
    if tp not in ("s1200", "fslr") or mesh not in ("inflated", "veryinflated"):
        raise ValueError("Invalid template type or mesh type!")

    if tp == "s1200":
        return con_path_list(__base__, FSLR['S1200_tp'][f's1200_{mesh}'])

    if tp == "fslr":
        return tpget('fsLR', density='32k', suffix=mesh)


def _get_sulcus_data():
    """
    Load and return the sulcus depth data for the S1200 template.

    Returns
    -------
    ndarray
        Sulcus depth data with medial wall removed.
    """
    sulcus = __base__ / FSLR['S1200_tp']['s1200_sulc']  # fsLR 59412
    sulcus = nib.load(sulcus).get_fdata().flatten()
    return reverse_mw(sulcus)


def _draw_colorbar_pdf(brain, low, up, legend, output, cmap, show=True):
    """
    Draw a colorbar on a PDF canvas alongside the brain image.

    Parameters
    ----------
    brain : str
        Path to the brain image file.
    low : float or str
        Lower bound of the color range.
    up : float or str
        Upper bound of the color range.
    legend : str
        Title for the legend/colorbar.
    output : str
        Output filename for the PDF.
    cmap : str
        Name of the colormap to use for the colorbar image.
    show : bool, optional
        Whether to display the generated PDF.
    """
    low = str(low)
    up = str(up)
    brain_size = {'width': 200, 'height': 850 / 6}
    colormap_size = {'width': 100, 'height': 10}
    canvas_size = {'width': brain_size['width'] + 40, 'height': brain_size['height']}

    c = canvas.Canvas(output, pagesize=(canvas_size['width'], canvas_size['height']))
    c.drawImage(brain, 0, 0, width=brain_size['width'], height=brain_size['height'],
                preserveAspectRatio=True, mask='auto')

    c.saveState()
    c.translate(brain_size['width'], brain_size['height'])
    c.rotate(-90)
    margin_colorbar_brain = 10

    c.drawImage(f"{__base__}/atlas/{cmap}_rev.png",
                brain_size['height'] / 2 - colormap_size['width'] / 2, margin_colorbar_brain,
                width=100, height=10, mask='auto')

    c.setFont("Helvetica", 8)
    c.drawCentredString(brain_size['height'] / 2, 25, legend)
    c.restoreState()

    c.setFont("Helvetica", 8)
    c.drawCentredString(brain_size['width'] + margin_colorbar_brain + colormap_size['height'] / 2,
                        (brain_size['height'] - colormap_size['width']) / 2 - 8, low)
    c.drawCentredString(brain_size['width'] + margin_colorbar_brain + colormap_size['height'] / 2,
                        (brain_size['height'] + colormap_size['width']) / 2 + 2, up)
    c.save()
    if show:
        display_pdf2(output)


def _paste_subcortex(cc, sc, hm):
    """
    Paste a subcortical image onto the cortical canvas.

    Parameters
    ----------
    cc : PIL.Image.Image
        The cortical canvas image.
    sc : str
        Path to the subcortical image file.
    hm : str
        Hemisphere ('L' or 'R').

    Returns
    -------
    PIL.Image.Image
        The combined image.
    """
    if hm == 'L':
        sc = Image.open(sc).resize((341, 285))
        cc.paste(sc, (273, 840), sc)
    else:
        sc = Image.open(sc).resize((341, 295))
        cc.paste(sc, (1058, 835), sc)
    return cc


def _combine_cortex_subcortex(cc, scl, scr, fig_name):
    """
    Combine cortical and subcortical images into a single figure.

    Parameters
    ----------
    cc : str
        Path to the cortical image file.
    scl : str
        Path to the left subcortex image file.
    scr : str
        Path to the right subcortex image file.
    fig_name : str
        Output filename for the combined image.
    """
    cc = Image.open(cc)
    if cc.size != (2000, 1181):
        raise ValueError('Background image not in the correct size!')

    cc = _paste_subcortex(cc, scl, 'L')
    cc = _paste_subcortex(cc, scr, 'R')
    cc.save(fig_name)


def _add_data_layers(brain, data, cmap, color_range, cbar_label=None, pn=False, sulc=False, sulcus_data=None, outline=False, hm=None, system=False):
    """
    Helper to add sulcus, data, pn, outline, and system layers to the brain plot.

    Parameters
    ----------
    brain : surfplot.Plot
        The brain plot object.
    data : ndarray
        The data to plot.
    cmap : str or object
        Colormap.
    color_range : tuple or list
        [min, max] color range.
    cbar_label : str, optional
        Label for the colorbar.
    pn : bool, optional
        Positive/Negative split flag.
    sulc : bool, optional
        Whether to add sulcus layer.
    sulcus_data : ndarray, optional
        Sulcus depth data.
    outline : bool, optional
        Whether to draw outlines.
    hm : str, optional
        Hemisphere ('lh' or 'rh').
    system : bool, optional
        Whether to add system boundaries.
    """
    if sulc and sulcus_data is not None:
        brain.add_layer(sulcus_data, cmap='binary_r', cbar=False)

    if pn:
        cr_lower, cr_upper = color_range
        cr_middle = (cr_lower + cr_upper) / 2
        brain.add_layer(data * (data < cr_middle), cmap=cmap, color_range=color_range,
                        zero_transparent=True, cbar_label=cbar_label)
        brain.add_layer(data * (data > cr_middle), cmap=cmap, color_range=color_range,
                        zero_transparent=True, cbar_label=cbar_label)
    else:
        brain.add_layer(data, cmap=cmap, color_range=color_range,
                        zero_transparent=True, cbar_label=cbar_label)

    if outline:
        color_list = ["#8C8C8C", "#8C8C8C"]
        n_fine = 1000
        just_black = get_cm(color_list, n_fine, 'black')
        brain.add_layer(data, cmap=just_black, zero_transparent=True, as_outline=True, cbar=False)
    
    if system:
        yeo7 = __base__ / 'atlas' / 'Yeo2011' / 'sf_value.npy'
        yeo7 = np.load(yeo7).astype(np.float64)
        yeo7 = a2w(yeo7)
        color_list = ["#000000", "#000000"]
        n_fine = 1000
        just_black = get_cm(color_list, n_fine, 'black')
        brain.add_layer(yeo7, cmap=just_black, zero_transparent=True, as_outline=True, cbar=False)


def _screenshot_and_trim(fig, fig_name, cmap, color_range, legend='legend', trim=True, if_cbar=True, show=True, save=True):
    """
    Helper to render, screenshot, trim, and save the figure with optional colorbar.

    Parameters
    ----------
    fig : surfplot.Plot
        The rendered figure object.
    fig_name : str
        Base name for output files.
    cmap : str or object
        Colormap.
    color_range : tuple or list
        [min, max] color range.
    legend : str, optional
        Legend title.
    trim : bool, optional
        Whether to trim whitespace.
    if_cbar : bool, optional
        Whether to include colorbar.
    show : bool, optional
        Whether to display the result.
    save : bool, optional
        Whether to save the result.
    """
    tmp_png = tmp_name('.png')
    fig.screenshot(tmp_png, transparent_bg=True)

    if trim:
        with wand_img(filename=tmp_png) as img:
            img.trim()
            img.save(filename=tmp_png)

    if if_cbar is False:
        shutil.copy(tmp_png, f"{fig_name}.png")
        Path(tmp_png).unlink(missing_ok=True)
    else:
        cmap_name = cmap
        if type(cmap_name) is not str:
            cmap_name = cmap_name.name
        _draw_colorbar_pdf(tmp_png, color_range[0], color_range[1],
                           legend, f"{fig_name}.pdf", cmap_name, show=show)
        Path(tmp_png).unlink(missing_ok=True)
        if save:
            with wand_img(filename=fig_name + '.pdf[0]', resolution=600) as img:
                img.format = 'png'
                img.save(filename=fig_name + '.png')
        else:
            Path(f"{fig_name}.pdf").unlink(missing_ok=True)


@deprecated(version='0.1.0', reason='Old codes, but dont have the heart to delete....')
def draw_hemi(
    data: np.ndarray,
    cmap: Union[str, object] = 'coolwarm',
    color_range: Optional[Union[List[float], Tuple[float, float]]] = None,
    layout: str = 'row',
    fig_name: str = 'brain',
    tp: str = 'fslr',
    mesh: str = 'veryinflated',
    trim: bool = True,
    pn: bool = False,
    cbar_label: Optional[str] = None,
    sulc: bool = False,
    outline: bool = False,
    system: bool = False,
    legend: str = 'legend',
    save: bool = True,
    show: bool = True,
    if_cbar: bool = True,
    force_nooutline: bool = False,
    hm: str = 'lh',
    just_mesh: bool = False,
) -> Plot:
    """
    Draw and save a heatmap for a single hemisphere.

    Parameters
    ----------
    data : ndarray
        Data to be plotted for the hemisphere.
    cmap : str or object, optional
        Colormap to use for data visualization. Default 'coolwarm'.
    color_range : list or tuple, optional
        [min, max] color range. If None, inferred from data.
    layout : str, optional
        Layout of the brain views ('grid', 'column', 'row'). Default 'row'.
    fig_name : str, optional
        Base name for output files. Default 'brain'.
    tp : str, optional
        Template type ('s1200' or 'fslr'). Default 'fslr'.
    mesh : str, optional
        Mesh inflation type ('inflated' or 'veryinflated'). Default 'veryinflated'.
    trim, pn, cbar_label, sulc, outline, system, legend, save, show, if_cbar, force_nooutline, hm, just_mesh : optional
        Other plotting options.

    Returns
    -------
    Plot
        The surfplot Plot object.
    """
    layout_key = f"{layout}_hm"
    if layout_key not in _LAYOUT_CONFIG:
        raise ValueError(f"Invalid layout: {layout}")
    config = _LAYOUT_CONFIG[layout_key]
    size = config['size']
    zoom = config['zoom']

    tp_files = _get_mesh_paths(tp, mesh)

    if hm == "lh":
        tp_surf = tp_files[0]
    elif hm == "rh":
        tp_surf = tp_files[1]
    else:
        raise ValueError("hm must be 'lh' or 'rh'")

    brain = Plot(tp_surf, layout=layout, size=size, zoom=zoom)

    sulcus_data = None
    if sulc:
        sulcus_raw = _get_sulcus_data()
        if hm == "lh":
            sulcus_data = sulcus_raw[:sulcus_raw.shape[0] // 2]
        elif hm == "rh":
            sulcus_data = sulcus_raw[sulcus_raw.shape[0] // 2:]

    _add_data_layers(brain, data, cmap, color_range, cbar_label, pn, sulc, sulcus_data, outline, hm, system)

    fig = brain.render()
    _screenshot_and_trim(fig, fig_name, cmap, color_range, legend, trim, if_cbar, show, save)

    return brain


def draw_cortex(
    data: np.ndarray,
    cmap: Union[str, object] = 'coolwarm',
    color_range: Union[List[float], Tuple[float, float]] = None,
    layout: str = 'grid',
    fig_name: str = 'brain',
    tp: str = 'fslr',
    mesh: str = 'veryinflated',
    trim: bool = True,
    pn: bool = False,
    cbar_label: Optional[str] = None,
    sulc: bool = False,
    outline: bool = False,
    system: bool = False,
    legend: str = 'legend',
    save: bool = True,
    show: bool = True,
    if_cbar: bool = True,
    force_nooutline: bool = False,
    hm: str = 'lh',
    just_mesh: bool = False,
) -> Plot:
    """
    Render cortical data.

    Parameters
    ----------
    data : ndarray
        Data to be plotted. Data without subcortex will be restored. Shape=(64984 | 59412,) or any length in atlas from Schaefer, Gordon.
    cmap : str or object, optional
        Colormap to use for data visualization. Can be a string in matplotlib or any matplotlib color map object.
    color_range : list or tuple
        Minimum and maximum values for color scaling. If None, inferred from data.
    layout : str, optional
        Layout of the brain views ('grid', 'column', 'row').
    legend : str, optional
        Title for the colorbar.
    fig_name : str, optional
        Base name for output files.
    tp : str, optional
        Template type ('s1200' or 'fslr').
    mesh : str, optional
        Mesh inflation type ('inflated' or 'veryinflated').
    trim : bool, optional
        Whether to trim whitespace from output images.
    pn : bool, optional
        If True, splits positive and negative values (hides 0/middle values).
    show : bool, optional
        Whether to display the result in a notebook.
    cbar_label : str, optional
        Label for the colorbar.
    system : bool, optional
        Whether to overlay Yeo 7-network system boundaries.
    sulc : bool, optional
        Whether to overlay sulcal depth information.
    outline : bool, optional
        Whether to draw outlines around data.
    force_nooutline : bool, optional
        Force disable outline even for parcel-based maps.

    Returns
    -------
    Plot
        The surfplot Plot object.
    """
    if len(data.flatten()) not in _FSLR_LEN + _ATLAS_LEN:
        raise ValueError(f"Data with length of {len(data.flatten())} not in valid length!")

    if layout not in _LAYOUT_CONFIG:
        raise ValueError(f"Invalid layout: {layout}")
    config = _LAYOUT_CONFIG[layout]
    size = config['size']
    zoom = config['zoom']

    tp_files = _get_mesh_paths(tp, mesh)
    brain = Plot(tp_files[0], tp_files[1], layout=layout, size=size, zoom=zoom)

    if color_range is None:
        if np.nanmin(data) == 0:
            color_range = [np.round(second_smallest(data), 2), np.round(np.nanmax(data), 2)]
        else:
            color_range = [np.round(np.nanmin(data), 2), np.round(np.nanmax(data), 2)]

    sulcus_data = _get_sulcus_data() if sulc else None

    if len(data) in _ATLAS_LEN:
        data = a2w(data)
        if not force_nooutline:
            outline = True

    no_mw = get_nomw_vertex_n()
    if len(data) in no_mw:
        data = reverse_mw(data)

    _add_data_layers(brain, data, cmap, color_range, cbar_label, pn, sulc, sulcus_data, outline, hm, system)
    
    fig = brain.render()
    _screenshot_and_trim(fig, fig_name, cmap, color_range, legend, trim, if_cbar, show, save)

    return brain


def draw_subcortex_tian(
    data: np.ndarray,
    cmap: Union[str, object] = 'coolwarm',
    color_range: Union[List[float], Tuple[float, float]] = None,
    fig_name: str = 'sub_tian',
    trim: bool = True,
    legend: str = 'legend',
    save: bool = True,
    show: bool = True,
    if_cbar: bool = True,
):
    """
    Render subcortical data using the Tian 2020 atlas.

    Parameters
    ----------
    data : ndarray
        Data to be plotted. Shape=(1032 | 432,).
    cmap : str or object, optional
        Colormap to use for data visualization. Can be a string in matplotlib or any matplotlib color map object.
    color_range : list or tuple, optional
        Minimum and maximum values for color scaling. If None, inferred from data.
    fig_name : str, optional
        Base name for output files.
    trim : bool, optional
        Whether to trim whitespace from output images.
    legend : str, optional
        Title for the colorbar.
    save : bool, optional
        Whether to save the figure.
    show : bool, optional
        Whether to display the result in a notebook.
    if_cbar : bool, optional
        Whether to include a colorbar.
    
    Returns
    -------
    None
    """
    scalar = ta2w(data)

    if color_range is None:
        color_range = [np.nanmin(scalar), np.nanmax(scalar)]

    for hm in ('l', 'r'):
        if hm == 'l':
            value = scalar[:len(scalar) // 2]
            cpos = [1.5, 0.5, -1]
        else:
            value = scalar[len(scalar) // 2:]
            cpos = [-10, 4, -6]
        mesh = pv.read(__base__ / 'atlas' / 'tian2020' / f'tian_{hm}h_smooth.vtk')

        plotter = pv.Plotter(off_screen=True)
        plotter.background_color = 'white'

        plotter.add_mesh(mesh, scalars=value, cmap=cmap,
                         clim=[color_range[0], color_range[1]])
        plotter.camera_position = cpos
        plotter.enable_parallel_projection()
        output_filename = f"{fig_name}_{hm}h.png"
        plotter.screenshot(output_filename)
        plotter.close()

    if trim:
        for hm in ('l', 'r'):
            with wand_img(filename=f"{fig_name}_{hm}h.png") as img:
                img.transparent_color(Color('white'), alpha=0.0)
                img.crop(width=600, height=500, gravity='center')
                img.trim()
                img.save(filename=f"{fig_name}_{hm}h_trim.png")


def draw_sftian(
    data: np.ndarray,
    cmap: Union[str, object] = 'coolwarm',
    color_range: Optional[Union[List[float], Tuple[float, float]]] = None,
    fig_name: str = 'brain.png',
    layout: str = 'grid',
    pn: bool = False,
    trim: bool = True,
    cbar_label: Optional[str] = None,
    tp: str = 'fslr',
    mesh: str = 'veryinflated',
    system: bool = False,
    sulc: bool = False,
    outline: bool = False,
    just_mesh: bool = False,
    legend: str = 'legend',
    save: bool = True,
    show: bool = True,
    if_cbar: bool = True,
    force_nooutline: bool = False,
    hm: str = 'lh',
):
    """
    Plot both surface (Schaefer atlas) and subcortical data (Tian atlas) and combine them.

    Parameters
    ----------
    data : ndarray
        Data to be plotted. Shape=(1032 | 432,).
    cmap : str or object, optional
        Colormap to use for data visualization. Can be a string in matplotlib or any matplotlib color map object.
    color_range : list or tuple, optional
        Minimum and maximum values for color scaling. If None, inferred from data.
    fig_name : str, optional
        Base name for output files.
    layout : str, optional
        Layout of the brain views ('grid', 'column', 'row').
    pn : bool, optional
        If True, splits positive and negative values (hides 0/middle values).
    trim : bool, optional
        Whether to trim whitespace from output images.
    cbar_label : str, optional
        Label for the colorbar.
    tp : str, optional
        Template type ('s1200' or 'fslr').
    mesh : str, optional
        Mesh inflation type ('inflated' or 'veryinflated').
    system : bool, optional
        Whether to overlay Yeo 7-network system boundaries.
    sulc : bool, optional
        Whether to overlay sulcal depth information.
    outline : bool, optional
        Whether to draw outlines around data.
    just_mesh : bool, optional
        If True, plot only the mesh without data.
    
    Returns
    -------
    Plot
        The surfplot Plot object.
    """
    if len(data) not in (1032, 432):
        raise ValueError(f"Input data should be in the length of 432 or 1032, while your data is of length {len(data)}!")

    data = data.astype(np.float64)
    cc = data[:len(data) - 32]  # cortex
    tian = data[len(data) - 32:]  # subcortex

    # Draw cortex
    draw_cortex(
        data=cc,
        cmap=cmap, color_range=color_range, layout=layout,
        fig_name='cc', tp=tp, mesh=mesh, trim=trim,
        pn=pn, cbar_label=cbar_label, sulc=sulc, outline=outline,
        system=system, legend=legend, save=True, show=False,
        if_cbar=if_cbar, force_nooutline=force_nooutline, hm=hm,
        just_mesh=just_mesh
    )

    # Draw subcortex
    draw_subcortex_tian(
        tian,
        cmap=cmap, color_range=color_range, fig_name='sub_tian', trim=True,
        legend=legend, save=True, show=show, if_cbar=if_cbar
    )

    out_name = fig_name
    if not out_name.endswith('.png'):
        out_name = out_name + '.png'

    _combine_cortex_subcortex('cc.png', 'sub_tian_lh_trim.png', 'sub_tian_rh_trim.png', out_name)

    display(IPyImage(out_name, width=350))

    cleanup_files(['sub_tian_lh_trim.png', 'sub_tian_rh_trim.png',
                         'sub_tian_lh.png', 'sub_tian_rh.png', 'cc.png', 'cc.pdf'])

    if not save:
        Path(out_name).unlink(missing_ok=True)
