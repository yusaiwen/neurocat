import nibabel as nib
import numpy as np
from .util import (_get_bm,
                   FSLR)
nib.imageglobals.logger.setLevel(40)


# GIFTI
def save_gii_hm(data, hm, fname) -> None:
    """
    Save as a gii shape file

    Parameters
    ----------
    data: np.array
        Data to save in gii.
    hm: {'L', 'R'}
        "L" for left hemisphere, "R" for right hemisphere.
    fname: str

    Returns
    ----------
    Nothing
    """
    # check hemisphere
    if hm not in ("L", "R"):
        raise ValueError("Wrong input for hemisphere.")

    # check data length
    if len(data) != FSLR['vertex_len']['hemimw']:
        raise ValueError(f"Wrong data length({len(data)})!")

    structure_dic = dict(L="CortexLeft",
                         R="CortexRight")
    structure = structure_dic.get(hm)

    gii = nib.gifti.gifti.GiftiImage()
    data = nib.gifti.gifti.GiftiDataArray(np.array(data, dtype=np.float32))

    # assign the structure(L,R hemisphere) for GIFTI
    gii.add_gifti_data_array(data)
    gii.meta = nib.gifti.gifti.GiftiMetaData(dict(AnatomicalStructurePrimary=structure))

    gii.to_filename(f"{fname}_hemi-{hm}.shape.gii")


# def save gii_label() -> None:


def save_gii(data, path) -> None:
    """ 
    Save as a gii shape file
    
    
    Parameters
    ----------
    data: np.array
        Data to save in gii.


    Returns
    ----------
    Nothing
    """

    # check data length
    if len(data) != fslr_info.fs32k_vertex.get('gii2'):
        raise ValueError(f"Wrong data length({len(data)})!")
    hemimw =  FSLR['vertex_len']['hemimw']
    lh = data[: hemimw]
    rh = data[hemimw:]

    save_gii_hm(lh, 'L', path)
    save_gii_hm(rh, 'R', path)
    
#def save gii_label() -> None:
    ### don't want write QAQ


# cifti

# def get_cii_nomedialmask(c):
#     brain_model = c.header.get_axis(1)
#     return brain_model.surface_mask
#
#
# def _get_cii_nomedial_index(c):
#     """
#
#     Returns
#     --------------
#     index: list of index of nomedial wall index. First element is for left hemisphere, second element is for right hemisphere.
#     """
#     brain_model = c.header.get_axis(1)
#     mask_lh = brain_model.name == "CIFTI_STRUCTURE_CORTEX_LEFT"
#     mask_rh = brain_model.name == "CIFTI_STRUCTURE_CORTEX_RIGHT"
#
#     index_lh = brain_model.vertex[mask_lh]  # 29696
#     index_rh = brain_model.vertex[mask_rh]  # 29716
#
#     return [index_lh, index_rh]


# def cii_64k(c):
#     data_59k = c.get_fdata().flatten()
#     index = _get_cii_nomedial_index(c)
#
#     data = [np.zeros(32492), np.zeros(32492)]
#     data[0][index[0]] = data_59k[0:29696]
#     data[1][index[1]] = data_59k[29696:59412]
#
#     return np.concatenate(data, axis=0)


def _gen_cii_head(hm) -> tuple:
    """
    Generate cifti head according to the hemisphere.
    """
    # if hm not in ('lh', 'rh', 'lhrh'): # since this method could be accessed outside. Hence, check the legality
    #     raise ValueError("Not legal value for hemisphere specification!")

    # scalar axis
    scalar_axis = nib.cifti2.cifti2_axes.ScalarAxis(
        ["ysw"])  # yes, that's my name. use my code and save my name into your data

    # brain model
    bm = _get_bm(hm)

    return (scalar_axis, bm)


def _len2hemi(data) -> str:
    """
    Judge the hemisphere from the length of the data.
    """
    data_len = len(data)
    fs32k_vertex = FSLR['vertex_len']

    if data_len not in [fs32k_vertex['L'], fs32k_vertex['R'], fs32k_vertex['LR']]:
        raise ValueError(f"Input data length({data_len}) doesn't align with any/both hemisphere!")
    return fslr_info.fs32k_vertex_inv.get(data_len)


def gen_cii(data) -> nib.Cifti2Image:
    """
    Generate scalar cifti2 object for eigher hemiphere or both hemisphere.
    Data shouldn't contain any vertex value for medial wall.
    """
    data = np.array(data)

    # determine if the data is one dimension
    ndim = len(data.shape)
    if ndim != 1:
        raise ValueError("Wrong dimension for input data to be saved as cifti.")

    # judge which hemisphere and generate the header
    hm = _len2hemi(data)
    header = _gen_cii_head(hm)

    # generate cifti2 object
    cii = nib.Cifti2Image(np.array([data]),  # just in (1, n) dimension ...
                          header)
    return cii
