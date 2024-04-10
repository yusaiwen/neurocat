import nibabel as nib
import numpy as np
from brainspace.utils.parcellation import map_to_labels
from pathlib import (Path, PosixPath)
from typing import Union
from deprecation import deprecated
import toml
import importlib.resources as pkg_resources


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


def _get_bm_from_s1200(hm=None) -> nib.cifti2.BrainModelAxis:
    """
    Read brain model from S1200's sulcus file. We read one important meta from S1200.sulc_MSMAll.32k_fs_LR.dscalar.nii.
    * vertices: indicates the index of each vertex in the geometry.

    Parameters
    ----------
    hm: {'L', 'R'}
        Hemisphere specification.

    Returns
    -------
    bm_out: nib.cifti2.BrainModelAxis
        Cifti's brain model object.
    """

    # if hm not in ('L', 'R'):
    #     raise ValueError("Not legal value for hemisphere specification!")

    bm_ref = nib.load(__base__ / FSLR['S1200_tp']['s1200_sulc']).header.get_axis(1)
    vertex = bm_ref.vertex
    nvertices = bm_ref.nvertices

    hmmw = FSLR['vertex_len']['hemimw']  # 32492
    lh = FSLR['vertex_len']['L']  # 29696

    if hm == 'L':
        vertex = vertex[:lh]
        structure_name = FSLR['hemi_name']['L']
        nvertices = hmmw

    elif hm == 'R':
        vertex = vertex[lh:]  # don't change lh to rh, I am right [by ysw]
        structure_name = FSLR['hemi_name']['R']
        nvertices = hmmw

    bm_out = nib.cifti2.BrainModelAxis.from_surface(vertices=vertex,
                                                    nvertex=nvertices,
                                                    name=structure_name)
    return bm_out


def _get_bm(hm: str) -> nib.cifti2.cifti2_axes.BrainModelAxis:
    """
    Get Cifti's brain model.

    Parameters
    ----------
    hm: {'L', 'R', 'LR'}
        Hemisphere specification.

    Returns
    -------
        : CIFTI's brain models.
    """
    # if hm not in ['lh', 'rh', 'lhrh']: # since this method could be accessed outside. Hence, check the legality
    #     raise ValueError("Not legal value for hemisphere specification!")
    if hm in ['L', 'R', 'LR']:
        return _get_bm_from_s1200(hm)
    else:
        return _get_bm_from_s1200('lh') + _get_bm_from_s1200('rh')


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
        atlas_d = f64k_59k(atlas_d)

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
    nm: ndarray, shape = (59412,)
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

    len_nm = len(nm)
    if len_nm != lhrh:
        raise ValueError(f'Wrong data length({len_nm})!')
    index_lh, index_rh = _get_fslr_vertex()

    out = (np.zeros(hemimw), np.zeros(hemimw))
    out[0][index_lh] = nm[:lh]
    out[1][index_rh] = nm[lh:]

    if hm:
        return out
    else:
        return np.concatenate(out)


def f64k_59k(mw, hm: bool = False) -> Union[np.ndarray, tuple]:
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


