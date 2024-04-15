import nibabel as nib
import numpy as np
from brainspace.utils.parcellation import map_to_labels
from pathlib import (Path, PosixPath)
from typing import Union
from deprecation import deprecated
import toml
import importlib.resources as pkg_resources
import os


# read two config file
__base__ = pkg_resources.files("neurocat")
with open(__base__ / 'fslr.toml', 'r') as f:
    FSLR = toml.load(f)
with open(__base__ / 'atlas.toml', 'r') as f:
    ATLAS = toml.load(f).get('atlas')

nib.imageglobals.logger.setLevel(40)  # fuck nibabel stupid output


# Atlas
def _atlas_npar(atlas: str, par: int) -> Path:
    atlas_dic = dict(
        schaefer2018 = __base__ / f"atlas/Schaefer2018/Schaefer2018_{par}Parcels_7Networks_order.dlabel.nii",
        yeo2011 = __base__ / f"atlas/Yeo2011_{par}Networks_N1000.dlabel.nii"
    )

    file = Path(atlas_dic[atlas])

    if file.exists():
        return file
    else:
        raise ValueError(f"Parcellation number of {par} is not supported for {atlas}.")


def get_atlas(atlas: str = None, par: int = None) -> Path:
    if atlas not in ATLAS['ATLAS']:
        raise ValueError('Not supported atlas or you incorrectly type.')

    file = Path(__base__ / ATLAS[atlas])
    if file.is_file():
        return file.absolute()
    else:  # directory
        return _atlas_npar(atlas, par)








# def get_cii_gii_data(cg: Union[nib.cifti2.cifti2.Cifti2Image,
#                                nib.gifti.gifti.GiftiImage]) -> np.ndarray:
#     """
#     Return the data of GIFTI or CIFTI file. Since nibabel use two different method to get the data of them.
#
#     Parameters
#     ----------
#     cg: nib.cifti2.cifti2.Cifti2Image or nib.gifti.gifti.GiftiImage
#         A Cifti2Image or GiftiImage object.
#
#     Returns
#     -------
#     data: np.ndarray
#         Cifti or GIFTI's data.
#     """
#
#     if type(cg) is nib.cifti2.cifti2.Cifti2Image:
#         return cg.get_fdata().flatten()
#     elif type(cg) is nib.gifti.gifti.GiftiImage:
#         return cg.agg_data().flatten()
#     else:
#         raise ValueError(f"Input not GIFTI or CIFTI object!")


def get_cii_gii_data(cg) -> np.ndarray:
    """
    Return the data of GIFTI or CIFTI file. Since nibabel use two different method to get the data of them.

    Parameters
    ----------
    cg: nib.cifti2.cifti2.Cifti2Image or nib.gifti.gifti.GiftiImage
        A Cifti2Image or GiftiImage object.

    Returns
    -------
    data: np.ndarray
        Cifti or GIFTI's data.
    """
    cg = Path(cg)

    if not cg.exists():
        raise FileNotFoundError(f"{cg} not found!")

    cg = nib.load(cg)
    if type(cg) is nib.cifti2.cifti2.Cifti2Image:
        return cg.get_fdata().flatten()
    elif type(cg) is nib.gifti.gifti.GiftiImage:
        return cg.agg_data().flatten()
    else:
        raise ValueError(f"Input not GIFTI or CIFTI object!")


def _atlas2array(atlas):
    """
    If the atlas is gifti/cifti file seperated in two files, this function will concatenated them into one array.
    If the atlas is just one cifti file, return the value of it.


    Parameters
    ----------
    atlas

    Returns
    -------

    """

    if type(atlas) in [tuple, list, np.array]:
        atlas = np.array(atlas)
        if len(atlas) == 1:
            atlas = atlas[0]
        elif len(atlas) == 2:
            atlas1 = nib.load(atlas[0]).get_fdata().flatten()
            # notice that the two hemispheres have same indeces for the same areas
            # Take, V1, left-V1 and right-V1 are labelled as 1
            atlas2 = nib.load(atlas[1]).get_fdata().flatten()
            atlas2[atlas2 >= 1] = atlas2[atlas2 >= 1] + atlas1.max()
            atlas = np.concatenate([atlas1, atlas2], axis=0)
        else:
            raise Exception("Atlas file outnumbers!!")
    elif type(atlas) is str:
        atlas = nib.load(atlas).get_fdata().flatten()
    else:
        raise Exception("Atlas unknown type!")
    return atlas


def _len2atlas(data_len: int) -> str:
    """
    Judge which atlas the data is based from the length of the data.

    Parameters
    ----------
    data_len: int
        The length of the data.

    Returns
    -------

    """
    schaefer = dict(zip(np.arange(100, 1001, 100),
                        ['schaefer2018'] * 10)
                    )
    yeo = dict(zip([7, 17],
                   ['yeo2011'] * 2)
               )
    no_par = {
        360: "glasser2016",
        333: "gorden2016"
    }

    atlas_dic = {**schaefer, **yeo, **no_par}
    if data_len not in atlas_dic.keys():
        raise ValueError(f"Not recognized data length:{data_len}")
    else:
        return atlas_dic[data_len]


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
    atlas_n = _len2atlas(data_len)  # name
    atlas_f = get_atlas(atlas_n, data_len)  # file
    atlas_d = get_cii_gii_data(atlas_f)  # data

    if len(atlas_d) == FSLR['vertex_len']['gii2']:
        atlas_d = f64k_2_59k(atlas_d)

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
    return f59k_2_64k(data_59k)


# to modify but not now
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
        out = f59k_2_64k(data, hm=True)

        if hm:
            return out
        else:
            return np.concatenate(out)
    else:
        raise ValueError(f"Wrong length({len_c}) of input file.")


@deprecated
def fetch_data(lh, rh, den='32k', p=Path()):
    """
    Give the function two hemisphere's path, concatinate them and unmask the medial wall to a left and right dictionary.
    """

    l_data, r_data = nib.load(lh).agg_data(), nib.load(rh).agg_data()

    return unmask_medial(l_data, r_data, den)


def thresh_array(data, threshold):
    """
    Threshold an numpy array, with
    * the value lower than the threshold being 0
    * the value greater than or equal to threshold being what it is

    Parameters
    -----------
    data: an numpy array(one demension) to be set a threhold

    threshold: a value(threshold)
    """
    return data * (data >= threshold)


@deprecated  # 这个改改还能用
def unmask_medial(lh, rh, den="32k", atlas="fsLR", threshold=float('-inf')):
    """
    unmask medial area

    lh: left hemisphere numpy array
    rh: right hemispehre numpy array
    den: resolution <32k|164k>
    """

    lh, rh = thresh_array(lh, threshold), thresh_array(rh, threshold)

    nomedialwall_L, nomedialwall_R = cii_2_64k(FSLR['S1200_tp']['s1200_medialwall'], True)

    return dict(left=np.multiply(lh, 1 - nomedialwall_L),
                right=np.multiply(rh, 1 - nomedialwall_R))


def _get_fslr_vertex() -> tuple:
    """
    Get fsLR cifti's 59k vertecies' indecies.

    Return
    -----------
    vertex_index: tuple
        vertex index for 59k seperated in a tuple.
        vertex_index[0] for left hemisphere, vertex_index[1] for right hemisphere.
    """
    lh = FSLR['vertex_len']['L']  # 29696
    dscalar_ref = __base__ / FSLR['S1200_tp']['s1200_sulc']
    bm_ref = nib.load(dscalar_ref).header.get_axis(1)
    vertex = bm_ref.vertex

    return vertex[:lh], vertex[lh:]


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



def con_path_list(path: PosixPath, ls: list) -> list:
    """
    Concatenate a paths with multiple child files as a list.

    Parameters
    ----------
    path: PosixPath
        A PosixPath object of the parent directory.
    ls: list
        A list of the child relative paths.

    Returns
    : list
        A list of the concatenated paths.
    -------

    """
    con = []
    for file in ls:
        con.append(path / file)
    return con

def _get_cii_nomedialmask(c):
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


def change_stupid_cii(cii):

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


def gen_gii_hm(data, hm) -> nib.GiftiImage:
    """
    Save as a gii file

    Parameters
    ----------
    data: np.ndarray, shape=(32492, n)
        Data to save in gii.
    hm: {'L', 'R'}
        "L" for left hemisphere, "R" for right hemisphere.

    Returns
    ----------
    Nothing
    """
    # check hemisphere
    if hm not in ("L", "R"):
        raise ValueError("Wrong input for hemisphere.")

    # check data length
    if len(data.shape) == 1:
        data_len = data.shape[0]
    elif len(data.shape) == 2:
        data_len = data.shape[0]

    if data_len != FSLR['vertex_len']['hemimw']:
        raise ValueError(f"Wrong data length({len(data)})!")

    structure_dic = dict(L="CortexLeft",
                         R="CortexRight")
    structure = structure_dic.get(hm)

    gii = nib.gifti.gifti.GiftiImage()
    data = nib.gifti.gifti.GiftiDataArray(np.array(data, dtype=np.float32))

    # assign the structure(L,R hemisphere) for GIFTI
    gii.add_gifti_data_array(data)
    gii.meta = nib.gifti.gifti.GiftiMetaData(dict(AnatomicalStructurePrimary=structure))

    return gii
    # gii.to_filename(f"{fname}_hemi-{hm}.shape.gii")


def _data_to_giiform(data: np.ndarray):
    shape = data.shape

    gii_values = np.array(list(FSLR['fslr_gii_vertex'].values()))
    gii_values2 = np.concatenate((gii_values, gii_values*2))

    if len(shape) == 1:  # as it is, 1D
        return data
    elif len(shape) == 2 and shape[0] == 1 and shape[1] in gii_values2:  # CIFTI's dsclaler mode
        return data.flatten()
    elif len(shape) == 2 and shape[0] in gii_values2:  # GIFTI form: vertex*time
        return data
    elif len(shape) == 2 and shape[1] in gii_values2:  # time * vertex
        return data.T


def _judge_data_type(data: np.ndarray) -> tuple:
    """
    Determine the data type is time series or scaler data.
    GIFTI's timeseries is in transcope form(vertex*time) for CIFTI(time*vertex).

    Parameters
    ----------
    data: np.ndarray
        Data to check.

    Returns
    : str {'series', 'scaler'}
        Data type.
    -------

    """
    if len(data.shape) == 2 and data.shape[0] == 1:
        dtype = 'scaler'
        data = data.flatten()
    elif len(data.shape) == 1:
        dtype = 'scaler'
    else:
        dtype = 'series'

    return dtype, data


def gen_gii_hm2(data: np.ndarray, hm: str = None) -> nib.GiftiImage:
    if hm not in ("L", "R"):
        raise ValueError("Wrong input for hemisphere.")

    gii_values = FSLR['fslr_gii_vertex'].values()
    # gii_values = np.array(list(FSLR['fslr_gii_vertex'].values())) * 2  # double for both hemisphere

    # whether data is time series data?
    dtype, data = _judge_data_type(data)
    data = _data_to_giiform(data)

    # new the object of gifti
    gii = nib.gifti.gifti.GiftiImage()

    # generate data
    data = nib.gifti.gifti.GiftiDataArray(np.array(data, dtype=np.float32))
    gii.add_gifti_data_array(data)

    # assign the structure(L,R hemisphere) for GIFTI
    structure_dic = dict(L="CortexLeft",
                         R="CortexRight")
    structure = structure_dic.get(hm)
    gii.meta = nib.gifti.gifti.GiftiMetaData(dict(AnatomicalStructurePrimary=structure))
    return gii

def gen_gii(data, hm: str = 'LR') -> tuple:
    """
    Save as a gii shape file

    Parameters
    ----------
    data: np.array, shape=({'4k', '8k', '32k', '164k'}, n)
        Data to save in gii. Length should be 64984(fslr+medial wall).
    hm : {'L', 'R', 'LR'}, optional
        Whether the data is one hemisphere. Default is 'LR'.

    Returns
    ----------
    :tuple
        GIFTI object of the input data
    """

    if isinstance(data, tuple):
        data = np.concatenate(data)

    # check data length
    gii_values = np.array(list(FSLR['fslr_gii_vertex'].values())) * 2  # double for both hemisphere

    if len(data.shape) == 2:
        if data.shape[0] not in gii_values:
            data = data.T # data in shape=(*k, n) n stands for time point
    data_len = data.shape[0]
    if data_len not in gii_values:
        raise ValueError(f"Wrong data length({len(data)})!")


    hemimw = FSLR['vertex_len']['hemimw']
    if len(data.shape) == 1:
        lh = data[:hemimw]
        rh = data[hemimw:]
    elif len(data.shape) == 2:
        lh = data[:hemimw, :]
        rh = data[hemimw:, :]

    return gen_gii_hm(lh, 'L'), gen_gii_hm(rh, 'R')

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
    data_64k = f59k_2_64k(data)
    # convert to gii object
    print(data_64k.shape)
    return gen_gii(data_64k)


def NestedDictValues(d):
  for v in d.values():
    if isinstance(v, dict):
      yield from NestedDictValues(v)
    else:
      yield v


def _judge_density(data: np.ndarray) -> str:

    dtype, data = _judge_data_type(data)
    if dtype == 'series':
        data_len = dtype.shape[1]
    if dtype == 'scaler':
        data_len = dtype.shape[0]

    den_list = tuple(NestedDictValues(FSLR['density_info']))

    if data_len not in den_list:
        raise ValueError(f"This is not any hemisphere!")



# def fuck_mw(data):

    # determine denisty




# if not isinstance(cii, (str, os.PathLike, nib.GiftiImage)):
#     raise TypeError("Input not a str or PathLike or GIFTI object!")