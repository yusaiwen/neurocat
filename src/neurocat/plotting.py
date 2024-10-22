# import the basic package
import os
from surfplot import Plot  # for plot surface
from templateflow.api import get as tpget
import pyvista as pv
from PIL import Image
from .color import *
from .util import (FSLR,
                   con_path_list,
                   __base__)
from .transfer import (
    reverse_mw,
    atlas_2_wholebrain as a2w,
    tian_a2w as ta2w
)
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

def draw_and_save_hm(layer1=None,
                  colorbar='coolwarm', color_range=None, hm="lh",
                  fig_name='brain', layout='row', pn=False, trim=True,
                  cbar_label=None,
                  tp='s1200', mesh='veryinflated',
                  sulc=False, outline=False,
                  just_mesh=False) -> Plot:
    zoom = 1.7
    tp = _get_mesh(tp, mesh)

    if layout == 'column':
        size = (600, 800)
    elif layout == 'row':
        size = (800, 400)
        zoom = 1.25

    if hm == "lh":
        tp = tp[0]
    elif hm == "rh":
        tp = tp[1]

    brain = Plot(tp, layout=layout, size=size, zoom=zoom)

    if sulc:  # add sulcus information if needed
        sulcus = _get_sulcus()
        if hm == "lh":
            sulcus = sulcus[:sulcus.shape[0]//2]
        elif hm == "rh":
            sulcus = sulcus[sulcus.shape[0]//2:]
        brain.add_layer(sulcus, cmap='binary_r', cbar=False)

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
    fig = brain.render()
    fig.screenshot(fig_name + ".png", transparent_bg=True)  # no colorbar

    if trim is True:
        os.system(f"convert {fig_name}.png -trim {fig_name}.png")

    return brain

def draw_and_save(layer1=None,
                  colorbar='coolwarm', color_range=None,
                  fig_name='brain', layout='grid', pn=False, trim=True,
                  cbar_label=None,
                  tp='fslr', mesh='veryinflated',
                  system=False,
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
    system: str, optional
        Which system's outline to draw. Default False to draw nothing
    sulc: bool, optional
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
        size = (1200, 850)
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
        color_list = ["#909090", "#909090"]
        n_fine = 1000
        just_black = get_cm(color_list, n_fine, 'black')
        brain.add_layer(layer1,
                        cmap=just_black,
                        zero_transparent=True,
                        as_outline=True,
                        cbar=False)
    if system:
        yeo7 = __base__ / 'atlas' / 'Yeo2011' / 'sf_value.npy'
        yeo7 = np.load(yeo7).astype(np.float64)
        yeo7 = a2w(yeo7)
        color_list = ["#000000", "#000000"]
        n_fine = 1000
        just_black = get_cm(color_list, n_fine, 'black')
        brain.add_layer(yeo7,
                        cmap=just_black,
                        zero_transparent=True,
                        as_outline=True,
                        cbar=False
                        )
    # fig = brain.build()
    # fig.show()

    fig = brain.render()
    fig.screenshot(fig_name + ".png", transparent_bg=True)  # no colorbar

    if trim is True:
        os.system(f"convert {fig_name}.png -trim {fig_name}.png")

    return brain



def draw_subcortex_tian(scalar, cmap='coolwarm', color_range=None,
                           fig_name='sub_tian', trim=False):
    scalar = ta2w(scalar)

    if color_range is None:
        color_range = [np.nanmin(scalar), np.nanmax(scalar)]
    # load mesh
    for hm in ('l', 'r'):
        if hm == 'l':
            value = scalar[:len(scalar) // 2]
            cpos = [1.5, 0.5, -1]
        else:
            value = scalar[len(scalar) // 2:]
            cpos = [-10, 4, -6]
        mesh = pv.read(__base__ / 'atlas' / 'tian2020'/ f'tian_{hm}h_smooth.vtk')
        mesh.plot(scalars=value,
                  cmap=cmap,
                  off_screen=True,
                  background='White',
                  clim=[color_range[0], color_range[1]],
                  parallel_projection=True,
                  cpos=cpos,
                  show_axes=False,
                  screenshot=f'{fig_name}_{hm}h.png'
                  )
        if trim is True:
            for hm in ('l', 'r'):
                os.system(f'convert {fig_name}_{hm}h.png  -transparent white -gravity center -crop 600x500+0+0 -trim {fig_name}_{hm}h_trim.png')


def _cc_add_hm(cc, sc, hm):
    if hm == 'L':
        sc = Image.open(sc).resize((258, 215))
        cc.paste(sc, (180, 578), sc)
    else:
        sc = Image.open(sc).resize((258, 224))
        cc.paste(sc, (722, 578), sc)
    return cc

def combine_cc_sc(cc, scl, scr, fig_name):
    cc = Image.open(cc)
    # make sure cc is in (1162, 822)
    if cc.size != (1162, 822):
        raise ValueError('Background image not in the correct size!')

    cc = _cc_add_hm(cc, scl, 'L')
    cc = _cc_add_hm(cc, scr, 'R')

    cc.save(fig_name)


def draw_sftian_save(layer1=None,
                   colorbar='RdBu_r', color_range=None,
                   fig_name='brain.png', layout='grid', pn=False, trim=True,
                   cbar_label=None,
                   tp='fslr', mesh='veryinflated',
                   system=False,
                   sulc=False, outline=False,
                   just_mesh=False):

    if len(layer1) != 432:
        raise ValueError('Input data should be in the length of 432, while your data is of length {len(layer1)}!')
    # take apart between subcortex and cerebral cortex
    layer1 = layer1.astype(np.float64)
    cc = layer1[:400]
    tian = layer1[400:]

    cc = a2w(cc)
    draw_and_save(layer1=cc,
                  colorbar=colorbar, color_range=color_range,
                  fig_name='cc', layout=layout, pn=pn, trim=trim,
                  cbar_label=cbar_label,
                  tp=tp, mesh=mesh,
                  system=system,
                  sulc=sulc, outline=outline,
                  just_mesh=just_mesh)

    draw_subcortex_tian(tian, colorbar, color_range,
                        trim=True)

    if not fig_name.endswith('.png'):
        fig_name = fig_name + '.png'
    combine_cc_sc('cc.png', 'sub_tian_lh_trim.png', 'sub_tian_rh_trim.png', fig_name)
    os.system('rm sub_tian_lh_trim.png sub_tian_rh_trim.png sub_tian_lh.png sub_tian_rh.png cc.png')