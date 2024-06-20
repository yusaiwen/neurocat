import numpy as np
import nibabel as nib
import pandas as pd
import os
from pathlib import Path
from deprecated.sphinx import deprecated
from typing import Union
from brainspace.utils.parcellation import map_to_labels
from neuromaps.transforms import fslr_to_fslr
from .util import (FSLR,
                   __base__,
                   judge_density,
                   get_cii_gii_data,
                   gen_gii_hm,gen_gii,
                   _data_to_ciiform, _data_to_npform, _data_to_giiform,
                   tmpname,
                   _get_bm,
                   len2atlas,
                   get_atlas)
import time


@deprecated(version='0.1.0', reason="Update to reverse_mw()")
def f59k_2_64k(nm: np.array, hm: bool = False) -> Union[np.ndarray, tuple]:
    """
    Add medial wall(all zero) to a no medial wall array.

    Parameters
    ----------
    nm: ndarray, shape = (n, 59412)
        no medial wall array
    hm: bool, default False
        Whether to separate each hemisphere data.
        If true, the first index of the output is left hemisphere,
        and the second index of the output is right hemisphere.

    Returns
    -------

    """
    lh = FSLR['vertex_len']['L']  # 29696
    lhrh = FSLR['vertex_len']['LR']  # 59412
    hemimw = FSLR['vertex_len']['hemimw']  # 32492

    if len(nm.shape) == 1:
        len_nm = len(nm)
    elif len(nm.shape) == 2:
        len_nm = nm.shape[1]
    if len_nm != lhrh:
        raise ValueError(f'Wrong data length({len_nm})!')
    index_lh, index_rh = _get_fslr_vertex()

    if len(nm.shape) == 1:
        out = (np.zeros(hemimw), np.zeros(hemimw))
        out[0][index_lh] = nm[:lh]
        out[1][index_rh] = nm[lh:]
    elif len(nm.shape) == 2:
        out = (np.zeros((nm.shape[0], hemimw)), np.zeros((nm.shape[0], hemimw)))
        out[0][:, index_lh] = nm[:, :lh]
        out[1][:, index_rh] = nm[:, lh:]
    if hm:
        return out
    else:
        if len(nm.shape) == 1:
            return np.concatenate(out)
        elif len(nm.shape) == 2:
            return np.concatenate(out, axis=1)

@deprecated(version='0.1.0', reason="Update to fuck_mw()")
def f64k_2_59k(mw, hm: bool = False) -> Union[np.ndarray, tuple]:
    if len(mw) != FSLR['vertex_len']['gii2']:
        raise  ValueError(f'Wrong data length({len(mw)})!')

    lh = FSLR['vertex_len']['L']  # 29696
    rh = FSLR['vertex_len']['R']  # 29716
    hemimw = FSLR['vertex_len']['hemimw']  # 32492
    index_lh, index_rh = _get_fslr_vertex()

    out = (np.zeros(lh), np.zeros(rh))
    out[0][:] = mw[:hemimw][index_lh]
    out[1][:] = mw[hemimw:][index_rh]

    if hm:
        return out
    else:
        return np.concatenate(out)




def fuck_mw(data: np.ndarray, hm=None) -> np.ndarray:
    """
    Mask the medial wall of a surface's data which contains the data for medial wall.

    Parameters
    ----------
    data: np.ndarray
        The data to be masked. If you feed a data with no need to be masked, I will warn you.
    hm: {'L', 'R'}, optional
        If data are from half hemisphere, the hemisphere should be spefication
        for two hemisphere are symmetric in vertex number. Default None.

    Returns
    -------
    : np.ndarray
        Masked data with no medial wall.
    """
    density, structure, data_len, data = judge_density(data)
    data = _data_to_ciiform(data)
    density_info = pd.read_csv(__base__ / 'S1200/fslr_vertex/density_info.csv')
    need_fuck_structure = ('hmmw', 'gii2')
    need_fuck_vertexn = density_info.query('structure in @need_fuck_structure')['vertex_n'].to_numpy()

    if data_len not in need_fuck_vertexn:
        if structure in ('MW', 'MWL', 'MWR'):
            raise ValueError('Medial wall slim to nothing?')
        return data

    mnw_index = np.load(__base__ / f'S1200/fslr_vertex/fslr-{density}_mw.npy')

    if structure == 'hmmw':  # only fuck half of the hemisphere
        if hm is None and density in ('32k', '164k'):
            raise ValueError(
                "Missing hemisphere spefication for the hemisphere with medial wall is symmetric across hemispheres!")
        elif hm is None and density in ('4k', '8k'):  # hm is not None or density in ('4k', '8k')
            hm = 'L'  # 4k 8k assymetry
        lh_n = density_info.query('density==@density and structure=="L"')['vertex_n'].values[0]
        rh_n = density_info.query('density==@density and structure=="R"')['vertex_n'].values[0]

        if hm == 'L':
            data_nmw = np.zeros((data.shape[0], lh_n))
            mnw_index = mnw_index[:lh_n]
        elif hm == 'R':
            data_nmw = np.zeros((data.shape[0], rh_n))
            hmmw = density_info.query('density==@density and structure=="hmmw"')['vertex_n'].values[0]
            mnw_index = mnw_index[lh_n:] - hmmw
        data_nmw[:,:] = data.T[mnw_index].T
    else:  # gii2
        LR = density_info.query('density==@density and structure=="LR"')['vertex_n'].values[0]
        data_nmw = np.zeros((data.shape[0], LR))
        data_nmw[:, :] = data.T[mnw_index].T

    return data_nmw


def reverse_mw(data: np.ndarray, hm=None) -> np.ndarray:
    """
    Reverse the medial wall of a surface's data which contains the data for medial wall.

    Parameters
    ----------
    data: np.ndarray
        The data to be masked. If you feed a data with no need to be masked, I will warn you.
    hm: {'L', 'R'}, optional
        If data are from half hemisphere, the hemisphere should be spefication
        for two hemisphere are symmetric in vertex number. Default None.

    Returns
    -------
    : np.ndarray
        Masked data with no medial wall.
    """
    # density, structure, data_len, data = judge_density(data)
    # density_info = pd.read_csv(__base__ / 'S1200/fslr_vertex/density_info.csv')
    # need_reverse_structure = ('L', 'R', 'LR')
    # need_reverse_vertexn = density_info.query('structure in @need_reverse_structure')['vertex_n'].to_numpy()
    #
    # if data_len not in need_reverse_vertexn:
    #     if structure in ('MW', 'MWL', 'MWR'):
    #         raise ValueError('Medial wall reverse to what?')
    #     return data
    # mnw_index = np.load(__base__ / f'S1200/fslr_vertex/fslr-{density}_mw.npy')
    #
    # if structure in ('L', 'R'):  # only reverse half of the hemisphere
    #     # if density in ('4k', '8k') and hm is None:  # 4k 8k ok syymetric
    #     #     raise ValueError("Missing hemisphere spefication for the hemisphere with medial wall is symmetric across hemispheres!")
    #     lh_n = density_info.query('density==@density and structure=="L"')['vertex_n'].values[0]
    #     hmmw = density_info.query('density==@density and structure=="hmmw"')['vertex_n'].values[0]
    #
    #     data_mw = np.zeros((data.shape[0], hmmw)) # may generate (1, vertex)
    #     data_mw.fill(np.nan)
    #
    #     if structure == 'L':  # 4k 8k ok syymetric
    #         data_mw[:, mnw_index[:lh_n]] = data
    #     else:
    #         data_mw[:, mnw_index[lh_n:] - hmmw] = data
    # else:  # LR
    #     gii2 = density_info.query('density==@density and structure=="gii2"')['vertex_n'].values[0]
    #     data_mw = np.zeros((data.shape[0], gii2))
    #     data_mw.fill(np.nan)
    #     data_mw[:, mnw_index] = data
    #
    # return _data_to_npform(data_mw)
    density, structure, data_len, data = judge_density(data)
    data = _data_to_ciiform(data)
    density_info = pd.read_csv(__base__ / 'S1200/fslr_vertex/density_info.csv')
    need_reverse_structure = ('L', 'R', 'LR')
    need_reverse_vertexn = density_info.query('structure in @need_reverse_structure')['vertex_n'].to_numpy()

    if data_len not in need_reverse_vertexn:
        if structure in ('MW', 'MWL', 'MWR'):
            raise ValueError('Medial wall reverse to what?')
        return data
    mnw_index = np.load(__base__ / f'S1200/fslr_vertex/fslr-{density}_mw.npy')

    if structure in ('L', 'R'):  # only reverse half of the hemisphere
        # if density in ('4k', '8k') and hm is None:  # 4k 8k ok syymetric
        #     raise ValueError("Missing hemisphere spefication for the hemisphere with medial wall is symmetric across hemispheres!")
        lh_n = density_info.query('density==@density and structure=="L"')['vertex_n'].values[0]
        hmmw = density_info.query('density==@density and structure=="hmmw"')['vertex_n'].values[0]
        data_mw = np.zeros((data.shape[0], hmmw)) # may generate (1, vertex)
        data_mw.fill(np.nan)

        if structure == 'L':  # 4k 8k ok syymetric
            data_mw[:, mnw_index[:lh_n]] = data
        else:
            data_mw[:, mnw_index[lh_n:] - hmmw] = data
    else:  # LR
        gii2 = density_info.query('density==@density and structure=="gii2"')['vertex_n'].values[0]
        data_mw = np.zeros((data.shape[0], gii2))  # (n, 59412)
        data_mw.fill(np.nan)
        data_mw[:, mnw_index] = data

    return _data_to_npform(data_mw)

def atlas_2_wholebrain_nm(data):
    """
    Convert atlas's short data to whole brain data without medial wall's vertex.
    Parameters
    ----------
    data: ndarray, shape=(n_atlas,)
        Scalar data for each atlas's ROI

    Returns
    ----------
    : ndarray, shape=(wholebrain_no-MW,)
        Array data of whole brain without medial wall's vertex.
    -------

    """

    data_len = len(data)
    atlas_n = len2atlas(data_len)  # name
    atlas_f = get_atlas(atlas_n, data_len)  # file
    atlas_d = get_cii_gii_data(atlas_f)  # data

    if len(atlas_d) == FSLR['vertex_len']['gii2']:
        atlas_d = fuck_mw(atlas_d).flatten()

    return map_to_labels(data, atlas_d, mask=(atlas_d!=0), fill=np.nan)


def atlas_2_wholebrain(data):
    """
    Convert atlas's short data to whole brain data with medial wall's vertex.
    Parameters
    ----------
    data: list-like
        Scalar data for each atlas's ROI

    Returns
    ----------
    : ndarray, shape=(wholebrain_has-MW,)
        Array data of whole brain with medial wall's vertex.
    -------

    """

    data_59k = atlas_2_wholebrain_nm(data)
    return reverse_mw(data_59k)

@deprecated(version='0.1.0', reason='has bug')
def cii_2_64k(cii, hm=False):
    """

    Parameters
    ----------
    cii: str or os.PathLike
        Path to the file to be transformed
    hm: bool, default False
        Whether to separate each hemisphere data.
        If true, the first index of the output is left hemisphere,
        and the second index of the output is right hemisphere.

    Returns
    -------

    """
    c = nib.load(cii)
    len_c = max(c.shape)
    data = c.get_fdata().flatten()

    if len_c == 64984:
        if hm is True:
            return [data[0:32492], data[32492:64984]]

        return c.get_fdata().flatten()

    elif len_c == 59412:
        out = reverse_mw(data, hm=True)

        if hm:
            return out
        else:
            return np.concatenate(out)
    else:
        raise ValueError(f"Wrong length({len_c}) of input file.")


def cii2gii(cii):
    """
    Convert an CIFTI file to GIFTI object.

    Parameters
    ----------
    cii: str or os.PathLike
        Path to CIFTI file.

    Returns
    -------

    """
    # check input type and existence
    if not isinstance(cii, (str, os.PathLike)):
        raise TypeError("Input not a str or PathLike!")
    if not Path(cii).exists():
        raise FileNotFoundError(f"{Path(cii).resolve()} not found!")

    # fetch 59k data
    data = cii_2_nmw(cii)

    # convert to 64k data
    data_64k = reverse_mw(data)
    # convert to gii object
    print(data_64k.shape)
    return gen_gii(data_64k)


def _get_cii_nomedialmask(c):
    """
    Get the no medial wall mask from a CIFTI file.

    Parameters
    ----------
    c: str or os.PathLike
        Path to a CIFTI file.

    Returns
    -------
    :np.ndarray
        Mask of no medial wall.

    """
    brain_model = c.header.get_axis(1)
    return brain_model.surface_mask


def _get_cii_nomedial_index(c):
    """

    Returns
    --------------
    index: list of index of nomedial wall index. First element is for left hemisphere, second element is for right hemisphere.
    """
    brain_model = c.header.get_axis(1)
    mask_lh = brain_model.name == "CIFTI_STRUCTURE_CORTEX_LEFT"
    mask_rh = brain_model.name == "CIFTI_STRUCTURE_CORTEX_RIGHT"

    index_lh = brain_model.vertex[mask_lh]  # 29696
    index_rh = brain_model.vertex[mask_rh]  # 29716

    return [index_lh, index_rh]

def cii_2_nmw(c) -> np.ndarray:
    """
    Given a CIFTI file, whose data length may not be standard as defined in FSLR(neigher in 29696, 29716, 59412, 32492, 64984).
    Differences are the medial wall area, but consistency comes in the no medial wall.
    Use this information to transfer the data into no medial wall.

    Parameters
    ----------
    c: os.path like or nib.cifti2.cifti2.Cifti2Image
        A CIFTI file whose data length may be strange(as MSC).

    Returns
    -------

    """
    if c is not nib.cifti2.cifti2.Cifti2Image:
        c = Path(c)
        if not c.exists():
            raise FileNotFoundError(f"{c.absolute()} not found!")

        c = nib.load(c)
    data_long = c.get_fdata() # (n, 64k) 64k may be larger for some stupid reason like MSC data
    index = _get_cii_nomedialmask(c)

    data = data_long[:, index]

    return data


def change_stupid_cii(cii) -> nib.Cifti2Image:
    """
    Stupid CIFTI has self-defined vertex number. This function will change the vertex number to norm.
    Parameters
    ----------
    cii

    Returns
    -------

    """
    cii_obj = nib.load(cii)
    data_len = cii_obj.get_fdata().shape[1]

    if data_len in FSLR['vertex_len'].values():
        raise ValueError("This is a fucking normal cifti file, not stupid!!!.")

    series = cii_obj.header.get_axis(0) # the as the original file is ok

    bm = _get_bm('LR') # change to normal
    header = (series, bm)

    data = cii_2_nmw(cii)

    # generate cifti2 object
    cii = nib.Cifti2Image(data,  # just in (1, n) dimension ...
                          header)
    return cii


def f2f(data: np.ndarray, tar_den: str, hm=None) -> np.ndarray:  #  no test
    if tar_den not in ('4k', '8k', '32k', '164k'):
        raise ValueError(f"Input target denisty({tar_den}) is not legal!")

    density, structure, data_len, data = judge_density(data)
    # acceptive scturecture: L, R, LR, hmmw, gii2
    # in which:
    ## L,R,LR need to reverse medial wall
    ## L,R,hmmw are single hemisphere
    ## hmmw needs to specify hemisphere
    if structure in ('MW', 'MWL', 'MWR'):  # three medial wall structure are excluded
        raise ValueError("This is not any hemisphere!")

    if structure == 'hmmw' and hm is None:
        raise ValueError('hmmw need to specify the hemisphere!')

    if density in ('4k', '8k') and structure in ('L', 'R') and hm is None:
        raise ValueError(
            '4k and 8k needs to specify hemisphere cuz they are symetric and cannot tell from the data length!')

    # reverse medial wall, for wb_command metric-resample accepts only GIFTI
    if structure in ('L', 'R', 'LR'):
        data = reverse_mw(data)

    # both hemi or single hemi?
    if structure in ('LR', 'gii2'):  # both hemisphere
        giis = gen_gii(data)
        giiklh_tpf = tmpname(".gii")
        giis[0].to_filename(giiklh_tpf)
        giikrh_tpf = tmpname(".gii")
        giis[1].to_filename(giikrh_tpf)
        time_end = time.time()  ##
        resampled = fslr_to_fslr((giiklh_tpf, giikrh_tpf), tar_den)
        os.remove(giiklh_tpf)
        os.remove(giikrh_tpf)

        resampled = _data_to_npform(np.concatenate((resampled[0].agg_data(), resampled[1].agg_data())))
        time_end = time.time()  ##
    else:  # single hemisphere
        if hm is None:
            hm = structure
        giis = gen_gii_hm(data, hm)
        gii_tpf = tmpname(".gii")
        giis.to_filename(gii_tpf)
        resampled = fslr_to_fslr(gii_tpf, tar_den, hm)
        os.remove(gii_tpf)

        resampled = _data_to_npform(resampled[0].agg_data())

    return resampled