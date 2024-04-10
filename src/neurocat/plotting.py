# import the basic package
import os
from surfplot import Plot  # for plot surface
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


def draw_and_save(layer1,
                  colorbar='RdBu_r', color_range=None,
                  fig_name='brain', layout='grid', trim=True,
                  cbar_label=None,
                  tp=None,
                  sulc=False, outline=False):
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

    if tp is None:
        tp = con_path_list(__base__, FSLR['S1200_tp']['s1200_veryinflated'])

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
