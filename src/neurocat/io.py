"""
Neuroimaging I/O utilities for handling GIFTI and CIFTI file formats.

This module provides functions to save and generate GIFTI shape files and CIFTI scalar/time series images,
primarily for cortical surface data in neuroimaging. It includes utilities for hemisphere-specific operations,
data validation, and file I/O with nibabel.
"""

import os
from pathlib import Path

import nibabel as nib
import numpy as np
from .util import (__base__,
                   FSLR,
                   gen_gii,
                   get_bm, gen_gii_hm,
                   judge_density)
from .transfer import change_stupid_cii

nib.imageglobals.logger.setLevel(40)


# GIFTI
def save_gii_hm(data, hm, fname) -> None:
    """Save as a gii shape file.

    Args:
        data (np.ndarray): Data to save in gii, shape=(32492,).
        hm ({'L', 'R'}): "L" for left hemisphere, "R" for right hemisphere.
        fname (str): Filename.

    Raises:
        FileNotFoundError: If parent directory does not exist.
    """
    if not Path(fname).parent.exists():
        raise FileNotFoundError(f"Path(fname)parent().resolve() does not exist!")

    gii = gen_gii_hm(data, hm)
    gii.to_filename(f"{fname}_hemi-{hm}.shape.gii")


def save_gii(data, fname) -> None:
    """Save as a gii shape file.

    Args:
        data (np.ndarray): Data to save in gii, shape=(64984,).
        fname (str or os.PathLike): Path to be saved for two hemispheres.

    Raises:
        FileNotFoundError: If parent directory does not exist.
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
    """Generate cifti head according to the hemisphere.

    Args:
        hm: Hemisphere specification.

    Returns:
        tuple: CIFTI header components.
    """
    # if hm not in ('L', 'R', 'LR'): # since this method could be accessed outside. Hence, check the legality
    #     raise ValueError("Not legal value for hemisphere specification!")

    # scalar axis
    scalar_axis = nib.cifti2.cifti2_axes.ScalarAxis(
        ["ysw"])  # yes, that's my name. use my code and save my name into your data

    # brain model
    bm = get_bm(hm)

    return (scalar_axis, bm)


# def _len2hemi(data) -> str:
#     """
#     Judge the hemisphere from the length of the data.
#     """
#
#     data_len = len(data)
#     fs32k_vertex = FSLR['vertex_len']
#
#     if data_len not in [fs32k_vertex['L'], fs32k_vertex['R'], fs32k_vertex['LR']]:
#         raise ValueError(f"Input data length({data_len}) doesn't align with any/both hemisphere!")
#     return fslr_info.fs32k_vertex_inv.get(data_len)


def gen_cii(data) -> nib.Cifti2Image:
    """Generate scalar cifti2 object for either hemisphere or both hemispheres.

    Data shouldn't contain any vertex value for medial wall.

    Args:
        data (np.ndarray): The data to be saved for CIFTI, shape=(n, 59412).

    Returns:
        nib.Cifti2Image: A Cifti image object.
    """
    data = np.array(data)

    # determine if the data is one dimension
    # ndim = len(data.shape)
    # if ndim != 1:
    #     raise ValueError("Wrong dimension for input data to be saved as cifti.")

    # judge which hemisphere and generate the header
    hm = judge_density(data)[1]
    header = _gen_cii_head(hm)

    # generate cifti2 object
    cii = nib.Cifti2Image(np.array([data]),  # just in (1, n) dimension ...
                          header)
    return cii


def save_cii(data: np.ndarray, fname=None) -> None:
    """Save CIFTI data.

    Args:
        data (np.ndarray): The data to be saved for CIFTI, shape=(59412,).
        fname (os.PathLike): File name of the CIFTI file.

    Raises:
        ValueError: If directory does not exist.
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
    """Generate the CIFTI's time series object.

    Args:
        data (np.ndarray): Time series to be saved for CIFTI, shape=(times, 59412).
        step (float): Sampling time (TR).
        start (float, optional): Starting time point. Defaults to 0.
        unit (str): Unit of the step size (one of ‘second’, ‘hertz’, ‘meter’, or ‘radian’).

    Returns:
        nib.Cifti2Image: CIFTI time series object.
    """
    size = data.shape[0]
    series_axis = nib.cifti2.cifti2_axes.SeriesAxis(start, step, size, unit)
    bm = get_bm('LR')
    header = (series_axis, bm)
    cii = nib.Cifti2Image(data, header)
    return cii

def save_cii_tm(data, fname, step: float, start: float=0, unit='SECOND'):
    """Save a CIFTI file time series data.

    Args:
        data: Time series data.
        fname (str): Path to save the CIFTI file. Should not include "dtseries.nii" in the end.
        start (float, optional): Starting time.
        step: Step size.
        unit: Unit.

    Raises:
        FileNotFoundError: If directory does not exist.
    """
    fname = Path(fname)
    if not fname.parent.exists():
        raise FileNotFoundError(f"Directory {fname.parent.absolute()} does not exist!")

    size = data.shape[0]
    cii = _gen_cii_tm(data, start, step, unit)
    cii.to_filename(f"{fname}.dtseries.nii")


def change_stupid_cii_save(cii, fname):
    """Convert some stupid CIFTI file whose data length is different from well-known length.

    Fortunately, no medial wall length is fixed.

    Args:
        cii: CIFTI input.
        fname: Filename.

    Raises:
        FileNotFoundError: If directory does not exist.
    """
    fname = Path(fname)
    if not fname.parent.exists():
        raise  FileNotFoundError(f"Directory {fname.parent.absolute()} does not exist!")

    cii_ojb = change_stupid_cii(cii)
    cii_ojb.to_filename(f"{fname}.dtseries.nii")