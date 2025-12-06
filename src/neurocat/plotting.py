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
                   display_pdf,
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
_ATLAS_LEN = tuple(range(100,1001,100)  # Schaefer
            ) + (998,  # Schaefer 100 fslr lack two labels
                 360,  # Glasser
            )

@dataclass
class PlotConfig:
    """
    Configuration for brain plotting functions.
    """
    # data: Union[np.ndarray, List[float]]
    cmap: Union[str, object] = 'coolwarm'
    color_range: Optional[Union[List[float], Tuple[float, float]]] = None
    layout: str = 'grid'
    fig_name: str = 'brain'
    tp: str = 'fslr'
    mesh: str = 'veryinflated'
    trim: bool = True
    pn: bool = False
    cbar_label: Optional[str] = None
    sulc: bool = False
    outline: bool = False
    system: bool = False
    legend: str = 'legend'
    save: bool = True
    show: bool = True
    if_cbar: bool = True
    force_nooutline: bool = False
    # Specific to single hemisphere or other functions
    hm: str = 'lh'
    just_mesh: bool = False


def _resolve_config(config: Optional[PlotConfig], **kwargs) -> PlotConfig:
    """
    Helper to resolve configuration from a PlotConfig object and keyword arguments.
    Kwargs override config fields.
    """
    # Filter kwargs to only those present in PlotConfig fields
    valid_fields = PlotConfig.__annotations__.keys()
    clean_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields and v is not None}
    
    if config is None:
        # Create new config with defaults, overridden by kwargs
        # We need to handle the case where kwargs might not cover all fields, 
        # so we instantiate default then replace, or just pass to constructor if we are sure.
        # Safer to instantiate default and replace.
        cfg = PlotConfig()
        return replace(cfg, **clean_kwargs)
    else:
        return replace(config, **clean_kwargs)


def _get_mesh_paths(tp, mesh) -> list:
    if tp not in ("s1200", "fslr") or mesh not in ("inflated", "veryinflated"):
        raise ValueError("Invalid template type or mesh type!")

    if tp == "s1200":
        return con_path_list(__base__, FSLR['S1200_tp'][f's1200_{mesh}'])

    if tp == "fslr":
        return tpget('fsLR', density='32k', suffix=mesh)


def _get_sulcus_data():
    sulcus = __base__ / FSLR['S1200_tp']['s1200_sulc']  # fsLR 59412
    sulcus = nib.load(sulcus).get_fdata().flatten()
    return reverse_mw(sulcus)


def _draw_colorbar_pdf(brain, low, up, legend, output, cmap, show=True):
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
        display_pdf(output)


def _paste_subcortex(cc, sc, hm):
    if hm == 'L':
        sc = Image.open(sc).resize((341, 285))
        cc.paste(sc, (273, 840), sc)
    else:
        sc = Image.open(sc).resize((341, 295))
        cc.paste(sc, (1058, 835), sc)
    return cc


def _combine_cortex_subcortex(cc, scl, scr, fig_name):
    cc = Image.open(cc)
    if cc.size != (2000, 1181):
        raise ValueError('Background image not in the correct size!')

    cc = _paste_subcortex(cc, scl, 'L')
    cc = _paste_subcortex(cc, scr, 'R')
    cc.save(fig_name)


def _add_data_layers(brain, data, cfg, sulcus_data=None):
    """
    Helper to add sulcus, data, pn, outline, and system layers to the brain plot.
    """
    if cfg.sulc and sulcus_data is not None:
        brain.add_layer(sulcus_data, cmap='binary_r', cbar=False)

    if cfg.pn:
        cr_lower, cr_upper = cfg.color_range
        cr_middle = (cr_lower + cr_upper) / 2
        brain.add_layer(data * (data < cr_middle), cmap=cfg.cmap, color_range=cfg.color_range,
                        zero_transparent=True, cbar_label=cfg.cbar_label)
        brain.add_layer(data * (data > cr_middle), cmap=cfg.cmap, color_range=cfg.color_range,
                        zero_transparent=True, cbar_label=cfg.cbar_label)
    else:
        brain.add_layer(data, cmap=cfg.cmap, color_range=cfg.color_range,
                        zero_transparent=True, cbar_label=cfg.cbar_label)

    if cfg.outline:
        color_list = ["#00000000", "#00000000"] if hasattr(cfg, 'hm') and cfg.hm in ('lh', 'rh') else ["#8C8C8C", "#8C8C8C"]
        n_fine = 1000
        just_black = get_cm(color_list, n_fine, 'black')
        brain.add_layer(data, cmap=just_black, zero_transparent=True, as_outline=True, cbar=False)
    
    if cfg.system:
        yeo7 = __base__ / 'atlas' / 'Yeo2011' / 'sf_value.npy'
        yeo7 = np.load(yeo7).astype(np.float64)
        yeo7 = a2w(yeo7)
        color_list = ["#000000", "#000000"]
        n_fine = 1000
        just_black = get_cm(color_list, n_fine, 'black')
        brain.add_layer(yeo7, cmap=just_black, zero_transparent=True, as_outline=True, cbar=False)


def _screenshot_and_trim(fig, fig_name, cfg):
    """
    Helper to render, screenshot, trim, and save the figure with optional colorbar.
    """
    tmp_png = tmp_name('.png')
    fig.screenshot(tmp_png, transparent_bg=True)

    if cfg.trim:
        with wand_img(filename=tmp_png) as img:
            img.trim()
            img.save(filename=tmp_png)

    if cfg.if_cbar is False:
        shutil.copy(tmp_png, f"{cfg.fig_name}.png")
        Path(tmp_png).unlink(missing_ok=True)
    else:
        cmap_name = cfg.cmap
        if type(cmap_name) is not str:
            cmap_name = cmap_name.name
        _draw_colorbar_pdf(tmp_png, cfg.color_range[0], cfg.color_range[1],
                           cfg.legend, f'{cfg.fig_name}.pdf', cmap_name, show=cfg.show)
        Path(tmp_png).unlink(missing_ok=True)
        if cfg.save:
            with wand_img(filename=cfg.fig_name + '.pdf[0]', resolution=600) as img:
                img.format = 'png'
                img.save(filename=cfg.fig_name + '.png')
        else:
            Path(f'{cfg.fig_name}.pdf').unlink(missing_ok=True)


def _cleanup_temp_files(file_list):
    """
    Helper to remove temporary files.
    """
    for f in file_list:
        Path(f).unlink(missing_ok=True)


class BrainPlotter:
    """
    Class to handle brain plotting operations, maintaining configuration state.
    """
    def __init__(self, config: Optional[PlotConfig] = None, **kwargs):
        self.config = _resolve_config(config, **kwargs)

    @deprecated(version='0.1.0', reason='Old codes')
    def draw_hemisphere(self, data=None, **kwargs) -> Plot:
        """
        Draw and save a heatmap for a single hemisphere.

        Args:
            data: 
            layout (str): Layout of the brain views ('grid', 'column', 'row').
            tp (str): Template type ('s1200' or 'fslr').
            hm (str): Hemisphere to plot ('lh' or 'rh').

        Returns:
            Plot: The surfplot Plot object.
        """
        # Set defaults specific to this function(different from others)
        kwargs.setdefault('layout', 'row')
        
        cfg = _resolve_config(self.config, **kwargs)

        layout_key = f'{cfg.layout}_hm'
        if layout_key not in _LAYOUT_CONFIG:
            raise ValueError(f'Invalid layout: {cfg.layout}')
        config = _LAYOUT_CONFIG[layout_key]
        size = config['size']
        zoom = config['zoom']
        
        tp_files = _get_mesh_paths(cfg.tp, cfg.mesh)

        if cfg.hm == "lh":
            tp_surf = tp_files[0]
        elif cfg.hm == "rh":
            tp_surf = tp_files[1]

        brain = Plot(tp_surf, layout=cfg.layout, size=size, zoom=zoom)

        sulcus_data = None
        if cfg.sulc:
            sulcus = _get_sulcus_data()
            if cfg.hm == "lh":
                sulcus_data = sulcus[:sulcus.shape[0] // 2]
            elif cfg.hm == "rh":
                sulcus_data = sulcus[sulcus.shape[0] // 2:]

        _add_data_layers(brain, data, cfg, sulcus_data)
        
        fig = brain.render()
        _screenshot_and_trim(fig, cfg.fig_name, cfg)

        return brain

    def draw_surface(self, data, **kwargs) -> Plot:
        """
        Render cortical data.

        Args:
            data (np.ndarray): Data to be plotted. Data without subcortex will be restored. Shape=(64984 | 59412,) or any length in atlas from Schaefer, Gordon.
            cmap (Union[str, object]): Colormap to use for data visualization. Can be a string in matplotlib or any matplotlib color map object.
            color_range (Optional[Union[List[float], Tuple[float, float]]]): Minimum and maximum values for color scaling. If None, inferred from data.
            layout (str): Layout of the brain views ('grid', 'column', 'row').
            legend (str): Title for the colorbar.
            fig_name (str): Base name for output files.
            tp (str): Template type ('s1200' or 'fslr').
            mesh (str): Mesh inflation type ('inflated' or 'veryinflated').
            trim (bool): Whether to trim whitespace from output images.
            pn (bool): If True, splits positive and negative values (hides 0/middle values).
            show (bool): Whether to display the result in a notebook.
            cbar_label (Optional[str]): Label for the colorbar.
            system (bool): Whether to overlay Yeo 7-network system boundaries.
            sulc (bool): Whether to overlay sulcal depth information.
            outline (bool): Whether to draw outlines around data.
            force_nooutline (bool): Force disable outline even for parcel-based maps.

        Returns:
            Plot: The surfplot Plot object.
        """
        
        cfg = _resolve_config(self.config, **kwargs)

        if len(data.flatten()) not in _FSLR_LEN + _ATLAS_LEN:
            raise ValueError(f'Data wiht length of {len(data.flatten())} not in valid length!')

        if cfg.layout not in _LAYOUT_CONFIG:
            raise ValueError(f'Invalid layout: {cfg.layout}')
        config = _LAYOUT_CONFIG[cfg.layout]
        size = config['size']
        zoom = config['zoom']

        tp_files = _get_mesh_paths(cfg.tp, cfg.mesh)
        brain = Plot(tp_files[0], tp_files[1], layout=cfg.layout, size=size, zoom=zoom)

        if cfg.color_range is None:
            if np.nanmin(data) == 0:
                color_range = [np.round(second_smallest(data), 2), np.round(np.nanmax(data), 2)]
            else:
                color_range = [np.round(np.nanmin(data), 2), np.round(np.nanmax(data), 2)]
            cfg = replace(cfg, color_range=color_range)

        sulcus_data = _get_sulcus_data() if cfg.sulc else None

        if len(data) in _ATLAS_LEN:
            data = a2w(data)
            cfg = replace(cfg, outline=not cfg.force_nooutline)

        no_mw = get_nomw_vertex_n()
        if len(data) in no_mw:
            data = reverse_mw(data)

        _add_data_layers(brain, data, cfg, sulcus_data)
        
        fig = brain.render()
        _screenshot_and_trim(fig, cfg.fig_name, cfg)

        return brain

    def draw_subcortex(self, data, **kwargs):
        """
        Render subcortical data using the Tian 2020 atlas.

        Args:
            data (np.ndarray): 
            cmap (Union[str, object]): Colormap to use for data visualization. Can be a string in matplotlib or any matplotlib color map object.
            color_range (Optional[Union[List[float], Tuple[float, float]]]): Minimum and maximum values for color scaling. If None, inferred from data.
            fig_name (str): Base name for output files.
            trim (bool): Whether to trim whitespace from output images.
        
        Returns:
            None
        """
        kwargs.setdefault('fig_name', 'sub_tian')
        kwargs.setdefault('trim', False)
        
        cfg = _resolve_config(self.config, **kwargs)
        
        scalar = ta2w(data) # 之后都改成data

        if cfg.color_range is None:
            color_range = [np.nanmin(scalar), np.nanmax(scalar)]
            cfg = replace(cfg, color_range=color_range)
        
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

            plotter.add_mesh(mesh, scalars=value, cmap=cfg.cmap,
                             clim=[cfg.color_range[0], cfg.color_range[1]])
            plotter.camera_position = cpos
            plotter.enable_parallel_projection()
            output_filename = f'{cfg.fig_name}_{hm}h.png'
            plotter.screenshot(output_filename)
            plotter.close()

        if cfg.trim:
            for hm in ('l', 'r'):
                with wand_img(filename=f"{cfg.fig_name}_{hm}h.png") as img:
                    img.transparent_color(Color('white'), alpha=0.0)
                    img.crop(width=600, height=500, gravity='center')
                    img.trim()
                    img.save(filename=f"{cfg.fig_name}_{hm}h_trim.png")

    def draw_tian2020(self, data, **kwargs):
        """
        Plot both surface (Scheafer atlas) and subcortical data (Tian atlas) and combine them.

        Args:
            data (np.ndarray): Data to be plotted. Shape=(1032 | 432,).
            cmap (Union[str, object]): Colormap to use for data visualization. Can be a string in matplotlib or any matplotlib color map object.
            color_range (Optional[Union[List[float], Tuple[float, float]]]): Minimum and maximum values for color scaling. If None, inferred from data.
            fig_name (str): Base name for output files.
            layout (str): Layout of the brain views ('grid', 'column', 'row').
            pn (bool): If True, splits positive and negative values (hides 0/middle values).
            trim (bool): Whether to trim whitespace from output images.
            cbar_label (Optional[str]): Label for the colorbar.
            tp (str): Template type ('s1200' or 'fslr').
            mesh (str): Mesh inflation type ('inflated' or 'veryinflated').
            system (bool): Whether to overlay Yeo 7-network system boundaries.
            sulc (bool): Whether to overlay sulcal depth information.
            outline (bool): Whether to draw outlines around data.
            just_mesh (bool): If True, plot only the mesh without data.
        
        Returns:
            Plot: The surfplot Plot object.
        """
        kwargs.setdefault('fig_name', 'brain.png')
        cfg = _resolve_config(self.config, **kwargs)

        if len(data) not in (1032, 432):
            raise ValueError(f'Input data should be in the length of 432 or 1032, while your data is of length {len(data)}!')
        
        data = data.astype(np.float64)
        cc = data[:len(data) - 32] # cortex
        tian = data[len(data) - 32:]  # subcortex

        # Draw cortex
        self.draw_surface(data=cc, fig_name='cc', show=False, **{k:v for k,v in kwargs.items() if k != 'fig_name'})

        # Draw subcortex
        self.draw_subcortex(tian, trim=True, **{k:v for k,v in kwargs.items() if k != 'fig_name'})

        fig_name = cfg.fig_name
        if not fig_name.endswith('.png'):
            fig_name = fig_name + '.png'
        
        _combine_cortex_subcortex('cc.png', 'sub_tian_lh_trim.png', 'sub_tian_rh_trim.png', fig_name)

        display(IPyImage(fig_name, width=350))

        _cleanup_temp_files(['sub_tian_lh_trim.png', 'sub_tian_rh_trim.png',
                             'sub_tian_lh.png', 'sub_tian_rh.png', 'cc.png'])


# Wrapper functions for backward compatibility
def draw_and_save_hm(data=None, config: Optional[PlotConfig] = None, **kwargs) -> Plot:
    """
    Draw and save a heatmap for a single hemisphere.

    Args:
        data: 
        config (PlotConfig, optional): Configuration object.
        layout (str): Layout of the brain views ('grid', 'column', 'row').
        tp (str): Template type ('s1200' or 'fslr').
        hm (str): Hemisphere to plot ('lh' or 'rh').

    Returns:
        Plot: The surfplot Plot object.
    """
    return BrainPlotter(config, **kwargs).draw_hemisphere(data)


def draw_and_save(data=None, config: Optional[PlotConfig] = None, **kwargs) -> Plot:
    """
    Render cortical data.
    
    Args:
        data (np.ndarray): Data to be plotted. Data without subcortex will be restored. Shape=(64984 | 59412,) or any length in atlas from Schaefer, Gordon.
        cmap (Union[str, object]): Colormap to use for data visualization. Can be a string in matplotlib or any matplotlib color map object.
        color_range (Optional[Union[List[float], Tuple[float, float]]]): Minimum and maximum values for color scaling. If None, inferred from data.
        layout (str): Layout of the brain views ('grid', 'column', 'row').
        legend (str): Title for the colorbar.
        fig_name (str): Base name for output files.
        tp (str): Template type ('s1200' or 'fslr').
        mesh (str): Mesh inflation type ('inflated' or 'veryinflated').
        trim (bool): Whether to trim whitespace from output images.
        pn (bool): If True, splits positive and negative values (hides 0/middle values).
        show (bool): Whether to display the result in a notebook.
        cbar_label (Optional[str]): Label for the colorbar.
        system (bool): Whether to overlay Yeo 7-network system boundaries.
        sulc (bool): Whether to overlay sulcal depth information.
        outline (bool): Whether to draw outlines around data.
        force_nooutline (bool): Force disable outline even for parcel-based maps.

    Returns:
        Plot: The surfplot Plot object.
    """
    return BrainPlotter(config, **kwargs).draw_surface(data)


def draw_subcortex_tian(data, config: Optional[PlotConfig] = None, **kwargs):
    """
    Render subcortical data using the Tian 2020 atlas.

    Args:
        data (np.ndarray): Data to be plotted. Shape=(1032 | 432,).
        cmap (Union[str, object]): Colormap to use for data visualization. Can be a string in matplotlib or any matplotlib color map object.
        color_range (Optional[Union[List[float], Tuple[float, float]]]): Minimum and maximum values for color scaling. If None, inferred from data.
        fig_name (str): Base name for output files.
        layout (str): Layout of the brain views ('grid', 'column', 'row').
        pn (bool): If True, splits positive and negative values (hides 0/middle values).
        trim (bool): Whether to trim whitespace from output images.
        cbar_label (Optional[str]): Label for the colorbar.
        tp (str): Template type ('s1200' or 'fslr').
        mesh (str): Mesh inflation type ('inflated' or 'veryinflated').
        system (bool): Whether to overlay Yeo 7-network system boundaries.
        sulc (bool): Whether to overlay sulcal depth information.
        outline (bool): Whether to draw outlines around data.
    """
    return BrainPlotter(config, **kwargs).draw_subcortex(data)


def draw_sftian_save(data=None, config: Optional[PlotConfig] = None, **kwargs):
    """
    Plot both surface (Scheafer atlas) and subcortical data (Tian atlas) and combine them.

    Args:
        data (np.ndarray): Data to be plotted. Shape=(1032 | 432,).
        cmap (Union[str, object]): Colormap to use for data visualization. Can be a string in matplotlib or any matplotlib color map object.
        color_range (Optional[Union[List[float], Tuple[float, float]]]): Minimum and maximum values for color scaling. If None, inferred from data.
        fig_name (str): Base name for output files.
        layout (str): Layout of the brain views ('grid', 'column', 'row').
        pn (bool): If True, splits positive and negative values (hides 0/middle values).
        trim (bool): Whether to trim whitespace from output images.
        cbar_label (Optional[str]): Label for the colorbar.
        tp (str): Template type ('s1200' or 'fslr').
        mesh (str): Mesh inflation type ('inflated' or 'veryinflated').
        system (bool): Whether to overlay Yeo 7-network system boundaries.
        sulc (bool): Whether to overlay sulcal depth information.
        outline (bool): Whether to draw outlines around data.
        just_mesh (bool): If True, plot only the mesh without data.
    
    Returns:
        Plot: The surfplot Plot object.
    """
    return BrainPlotter(config, **kwargs).draw_tian2020(data)


def plot_mesh(tp='fslr', mesh='veryinflated', hm='L', fig_name='brain'):
    """
    Plot brain mesh only for demonstration purposes.
    """
    tp_files = _get_mesh_paths(tp, mesh)
    if hm == 'L':
        brain = Plot(tp_files[0], size=(1200, 425), zoom=1.25)
    else:
        brain = Plot(surf_rh=tp_files[1], size=(1200, 425), zoom=1.25)

    fig = brain.render()
    tmp_png = tmp_name('.png')
    fig.screenshot(tmp_png, transparent_bg=True)
    
    with wand_img(filename=tmp_png) as img:
        img.trim()
        img.save(filename=f"{fig_name}.png")
    
    Path(tmp_png).unlink(missing_ok=True)
    return brain
