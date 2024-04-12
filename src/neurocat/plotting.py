# import the basic package
import os
from surfplot import Plot  # for plot surface
from templateflow.api import get as tpget
from .color import *
from .util import *
from .util import (FSLR,
                   cii_2_64k,
                   con_path_list,
                   __base__)


nib.imageglobals.logger.level = 40

# def fetch_tp(atlas="fsLR", den='32k', mesh='veryinflated', hcp=None, p=Path()):
#     if hcp is not None:
#         lh, rh = "S1200.L.very_inflated_MSMAll.32k_fs_LR.surf.gii", "S1200.R.very_inflated_MSMAll.32k_fs_LR.surf.gii"
#         return dict(left=lh,
#                     right=rh)
#     return fetch_atlas(atlas, den).get(mesh)


def _get_mesh(tp="s1200", mesh="veryinflated") -> list:
    """
    Get surface geometrical mesh.

    Parameters
    ----------
    tp: {"s1200", "fslr"}, default "s1200"
        Type of surface geometrical mesh.
    mesh: {"veryinflated", "inflated"}, default "veryinflated"
        Type of mesh.

    Returns
    -------
    : list
        Surface geometrical mesh.
    """

    if tp not in ("s1200", "fslr") or mesh not in ("inflated", "veryinflated"):
        raise ValueError("Invalid template type or mesh type!")

    if tp == "s1200":
        return con_path_list(__base__, FSLR['S1200_tp'][f's1200_{mesh}'])

    if tp == "fslr":
        return tpget('fsLR',
                     density='32k',
                     suffix=mesh)


def draw_and_save(layer1,
                  colorbar='RdBu_r', color_range=None,
                  fig_name='brain', layout='grid', trim=True,
                  cbar_label=None,
                  tp='s1200', mesh='veryinflated',
                  sulc=False, outline=False):
    """
    Plot brain surfaces with data layers

    Parameters
    ----------
    layer1: np.ndarray, shape=(64984,)
        Data to be plotted.
    colorbar: matplotlib colormap name or object, optional
        Colormap to use for data, default: 'RdBu_r'
    color_range: tuple[float, float], optional
        Minimum and maximum value for color map. If None, then the minimum
        and maximum values in `layer1` are used. Default: None
    fig_name: str, optional
        Name of output figure. Default: 'brain'.
    layout: {'grid', 'column', 'row'}, optional
        Layout in which to plot brain surfaces. 'row' plots brains as a
        single row ordered from left-to-right hemispheres (if applicable),
        'column' plots brains as a single column descending from
        left-to-right hemispheres (if applicable). 'grid' plots surfaces
        as a views-by-hemisphere (left-right) array; if only one
        hemipshere is provided, then 'grid' is equivalent to 'row'. By
        default 'grid'.
    trim: bool, optional
        If True, trim the output figure by ImageMagick's `convert` command.
        Defaults True.
    cbar_label: str, optional
            Label to include with colorbar if shown. Note that this is not
            required for the colorbar to be drawn. Default: None
    tp: {"s1200", "fslr"}, optional.
        Template type. Default "s1200"
    mesh: {"inflated", "veryinflated"}, optional
        Mesh type. Default "veryinflated"
    sulcL bool, optional
        Whether to plot suculs inforamtion. Default: False.
    outline: bool, optional
            Plot only an outline of contiguous vertices with the same value.
            Useful if plotting regions of interests, atlases, or discretized
            data. Not recommended for continous data. Default: False

    Returns
    -------

    """
    zoom = 1.7
    if layout == 'grid':
        size = (1200, 800)
    elif layout == 'column':
        size = (600, 1600)
    elif layout == 'row':
        size = (1600, 400)
        zoom = 1.25
    else:
        raise ValueError('layout must be "grid" or "column" or "row".')

    tp = _get_mesh(tp, mesh)

    brain = Plot(tp[0], tp[1], layout=layout, size=size, zoom=zoom)

    if sulc:
        sulcus = __base__ / FSLR['S1200_tp']['s1200_sulc'] # fsLR 59412
        brain.add_layer(cii_2_64k(sulcus), cmap='binary_r', cbar=False)

    brain.add_layer(layer1,
                    cmap=colorbar,
                    color_range=color_range,
                    zero_transparent=True,
                    cbar_label=cbar_label)
    if outline:
        color_list = ["#8C8C8C", "#8C8C8C"]
        n_fine = 1000
        just_black = get_cm(color_list, n_fine, 'black')
        brain.add_layer(layer1,
                        cmap=just_black,
                        zero_transparent=True,
                        as_outline=True,
                        cbar=False)
    # fig = brain.build()
    # fig.show()

    fig = brain.render()
    fig.screenshot(fig_name + ".png", transparent_bg=True)  # no colorbar

    if trim is True:
        os.system(f"convert {fig_name}.png -trim {fig_name}.png")

    return brain
