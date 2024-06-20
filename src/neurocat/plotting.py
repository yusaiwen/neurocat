# import the basic package
import os
from surfplot import Plot  # for plot surface
from templateflow.api import get as tpget
from .color import *
from .util import (FSLR,
                   con_path_list,
                   __base__)
from .transfer import (reverse_mw)
import nibabel as nib
nib.imageglobals.logger.setLevel(40)

# def fetch_tp(atlas="fsLR", den='32k', mesh='veryinflated', hcp=None, p=Path()):
#     if hcp is not None:
#         lh, rh = "S1200.L.very_inflated_MSMAll.32k_fs_LR.surf.gii", "S1200.R.very_inflated_MSMAll.32k_fs_LR.surf.gii"
#         return dict(left=lh,
#                     right=rh)
#     return fetch_atlas(atlas, den).get(mesh)


def _get_mesh(tp="fslr", mesh="veryinflated") -> list:
    """
    Get surface geometrical mesh.

    Parameters
    ----------
    tp: {"s1200", "fslr"}, default "fslr"
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

def _get_sulcus():
    sulcus = __base__ / FSLR['S1200_tp']['s1200_sulc']  # fsLR 59412
    sulcus = nib.load(sulcus).get_fdata().flatten()
    return reverse_mw(sulcus)

def draw_and_save(layer1=None,
                  colorbar='RdBu_r', color_range=None,
                  fig_name='brain', layout='grid', pn=False, trim=True,
                  cbar_label=None,
                  tp='s1200', mesh='veryinflated',
                  sulc=False, outline=False,
                  just_mesh=False) -> Plot:
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
    pn: bool, optional
        If True, Delete the zero. Defaults False.
    cbar_label: str, optional
        Label to include with colorbar if shown. Note that this is not
        required for the colorbar to be drawn. Default: None
    tp: {"s1200", "fslr"}, optional.
        Template type. Default "fslr"
    mesh: {"inflated", "veryinflated"}, optional
        Mesh type. Default "veryinflated"
    sulcL bool, optional
        Whether to plot suculs inforamtion. Default: False.
    outline: bool, optional
        Plot only an outline of contiguous vertices with the same value.
        Useful if plotting regions of interests, atlases, or discretized
        data. Not recommended for continous data. Default: False
    just_mesh: book, optional
        Return just the mesh. For those who want to add layer mannually.
        Default: False
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
        sulcus = _get_sulcus()
        brain.add_layer(sulcus, cmap='binary_r', cbar=False)

    if just_mesh:
        return brain

    if pn:
        cr_lower, cr_upper = color_range
        cr_middle = (cr_lower + cr_upper) / 2

        brain.add_layer(layer1*(layer1 < cr_middle),
                        cmap=colorbar,
                        color_range=color_range,
                        zero_transparent=True,
                        cbar_label=cbar_label)
        brain.add_layer(layer1 * (layer1 > cr_middle),
                        cmap=colorbar,
                        color_range=color_range,
                        zero_transparent=True,
                        cbar_label=cbar_label)
    else:
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
