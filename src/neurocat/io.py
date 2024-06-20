import os
from pathlib import Path

import nibabel as nib
import numpy as np
from .util import (__base__,
                   FSLR,
                   gen_gii,
                   _get_bm, gen_gii_hm,
                   judge_density,
                   _get_bm_from_s1200)
from .transfer import change_stupid_cii

nib.imageglobals.logger.setLevel(40)


# GIFTI
def save_gii_hm(data, hm, fname) -> None:
    """
    Save as a gii shape file

    Parameters
    ----------
    data: np.ndarray, shape=(32492, )
        Data to save in gii.
    hm: {'L', 'R'}
        "L" for left hemisphere, "R" for right hemisphere.
    fname: str

    Returns
    ----------
    Nothing
    """
    if not Path(fname).parent.exists():
        raise FileNotFoundError(f"Path(fname)parent().resolve() does not exist!")

    gii = gen_gii_hm(data, hm)
    gii.to_filename(f"{fname}_hemi-{hm}.shape.gii")


# def save gii_label() -> None:





def save_gii(data, fname) -> None:
    """
    Save as a gii shape file


    Parameters
    ----------
    data: np.array, shape=(64984,)
        Data to save in gii. Length should be 64984(fslr+medial wall).

    fname: str or os.Pathlike
        Path to be saved for two hemispheres.
    Returns
    ----------
    Nothing
    """
    giis = gen_gii(data)
    giis[0].to_filename(f"{fname}_hemi-L.shape.gii")
    giis[1].to_filename(f"{fname}_hemi-R.shape.gii")


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
    # if hm not in ('L', 'R', 'LR'): # since this method could be accessed outside. Hence, check the legality
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

    Parameters
    ----------
    data: np.ndarray, shape=(59412, )
        The data to be saved for CIFTI.

    Returns
    -------
    : nib.Cifti2Image
        A Cifti image object.
    """
    data = np.array(data)

    # determine if the data is one dimension
    ndim = len(data.shape)
    if ndim != 1:
        raise ValueError("Wrong dimension for input data to be saved as cifti.")

    # judge which hemisphere and generate the header
    hm = judge_density(data)[1]
    header = _gen_cii_head(hm)

    # generate cifti2 object
    cii = nib.Cifti2Image(np.array([data]),  # just in (1, n) dimension ...
                          header)
    return cii

def save_cii(data: np.ndarray, fname=None) -> None:
    """

    Parameters
    ----------
    data: np.ndarray, shape=(59412, )
        The data to be saved for CIFTI.
    fname: os.path like
        file name of the CIFTI file.

    """
    cii = gen_cii(data)

    fname = Path(fname)

    if not fname.parent.exists():
        raise ValueError(f"Directory {fname.parent.absolute()} already exists!")
    cii.to_filename(fname)


# def _gen_cii_tm_head(data, step: float, start: float=0, unit='SECOND') -> tuple:
#     """
#     Generate series CIFTI's head.
#
#     Parameters
#     ----------
#     start
#     step
#     size
#     unit
#
#     Returns
#     -------
#
#     """
#
#     size = data.shape[0]
#     series_axis = nib.cifti2.cifti2_axes.SeriesAxis(start, step, size, unit)
#     bm = _get_bm('LR')
#     return (series_axis, bm)


def _gen_cii_tm(data, step: float, start: float=0, unit='SECOND'):
    """
    Generate the CIFTI's time series object.

    Parameters
    ----------
    data: np.ndarray, shape=(times, 59412)
        Time series to be saved for CIFTI.
    step: float
        sampling time (TR)
    start: float, optional
        starting time point. Defaults to 0.
    unit: str
        Unit of the step size (one of ‘second’, ‘hertz’, ‘meter’, or ‘radian’)

    Returns
    : nib.Cifti2Image
        CIFTI time series object.
    -------

    """
    size = data.shape[0]
    series_axis = nib.cifti2.cifti2_axes.SeriesAxis(start, step, size, unit)
    bm = _get_bm('LR')
    header = (series_axis, bm)
    cii = nib.Cifti2Image(data, header)
    return cii

def save_cii_tm(data, fname, step: float, start: float=0, unit='SECOND'):
    """
    Save a CIFTI file time series data.

    Parameters
    ----------
    data
    fname: str
        Path to save the CIFTI file. Should not include "dtseries.nii" in the end.
    start: float, optional

    step
    unit

    Returns
    -------

    """

    fname = Path(fname)
    if not fname.parent.exists():
        raise FileNotFoundError(f"Directory {fname.parent.absolute()} does not exist!")

    size = data.shape[0]
    cii = _gen_cii_tm(data, start, step, unit)
    cii.to_filename(f"{fname}.dtseries.nii")


def change_stupid_cii_save(cii, fname):
    """
    Convert some stupid CIFTI file whose data length is different well-knwon length.
    Fortunately, no medial wall length is fixed.

    Parameters
    ----------
    cii
    fname

    Returns
    -------

    """
    fname = Path(fname)
    if not fname.parent.exists():
        raise  FileNotFoundError(f"Directory {fname.parent.absolute()} does not exist!")

    cii_ojb = change_stupid_cii(cii)
    cii_ojb.to_filename(f"{fname}.dtseries.nii")