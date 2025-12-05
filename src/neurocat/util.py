import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import (Path, PosixPath)
from typing import Union
from deprecated.sphinx import deprecated
import toml
import importlib.resources as pkg_resources
import os
import tempfile
from IPython.display import SVG, display, Image

# read two config file
__base__ = Path(pkg_resources.files("neurocat"))
with open(__base__ / 'fslr.toml', 'r') as f:
    FSLR = toml.load(f)
with open(__base__ / 'atlas.toml', 'r') as f:
    ATLAS = toml.load(f).get('atlas')

nib.imageglobals.logger.setLevel(40)  # fuck nibabel stupid output


def logger():
    import logging
    from rich.logging import RichHandler

    logging.basicConfig(
        level=eval("logging.INFO"),
        format="%(message)s",
        datefmt="[%m/%d/%Y %X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    log = logging.getLogger("rich")
    return log


# Create a temporary file in /dev/shm
def tmp_name(suffix=None, prefix=None):
    """
   Create a temporary file. If the operating system is Linux, the file should be in /dev/shm, where the files are stored in memory.
   Instead in Windows, the file

    Parameters
    ----------

    suffix : str
        Suffix of created filename
    prefix: str
    Returns
    -------
    fn : str
        Temporary filename; user is responsible for deletion
    """
    import platform
    if platform.system() == 'Linux':
        tmp_dir = '/dev/shm'
    if platform.system() == 'Darwin':  #  macos
        tmp_dir = None
    else: # windows
        tmp_dir = '.'

    # Create a temporary file in /dev/shm
    fd, fn = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=tmp_dir)
    os.close(fd) 

    return Path(fn)


def display_pdf(pdf):
    svg_tmp = tmp_name('.svg')
    os.system(f'pdf2svg {pdf} {svg_tmp}')
    display(SVG(filename=svg_tmp))
    Path(svg_tmp).unlink(missing_ok=True)


def display_pdf2(pdf):
    png_tmp = tmp_name('.png')
    os.system(f'convert -density 300 {pdf} {png_tmp}')
    Image(filename=png_tmp)
    Path(png_tmp).unlink(missing_ok=True)


# Atlas
def _atlas_npar(atlas: str, par: int) -> Path:
    if par == 998:  # schaefer 998 is 1000p in CIFTI version
        par = 1000
    atlas_dic = dict(
        schaefer2018=__base__ / f"atlas/Schaefer2018/Schaefer2018_{par}Parcels_7Networks_order.dlabel.nii",
        yeo2011=__base__ / f"atlas/Yeo2011_{par}Networks_N1000.dlabel.nii"
    )

    file = Path(atlas_dic[atlas])

    if file.exists():
        return file
    else:
        raise ValueError(f"Parcellation number of {par} is not supported for {atlas}.")


def get_atlas(atlas: str, par: int = None) -> Path:
    """
    Get a atlas's path by the name and parcel number.

    Parameters
    ----------
    atlas: str
        The name of the atlas.
    par: int
        Parcellation number of the atlas.

    Returns
    -------

    """
    if atlas not in ATLAS['ATLAS']:
        raise ValueError('Not supported atlas or you incorrectly type.')

    file = Path(__base__ / ATLAS[atlas])
    if file.is_file():
        return file.absolute()
    else:  # directory
        return _atlas_npar(atlas, par)


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
    if hm in ['L', 'R']:
        return _get_bm_from_s1200(hm)
    else:  # , 'LR'
        return _get_bm_from_s1200('L') + _get_bm_from_s1200('R')


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


def len2atlas(data_len: int) -> str:
    """
    Judge which atlas the data is based from the length of the data.

    Parameters
    ----------
    data_len: int
        The length of the data.

    Returns
    -------

    """
    atlas_df = pd.read_csv(__base__ / 'atlas/atlas.csv')
    # convert this dataframe to a dictionary
    atlas_dic = dict(zip(atlas_df['n_par'], atlas_df['atlas']))
    if data_len not in atlas_dic.keys():
        raise ValueError(f"Not recognized data length:{data_len}")
    else:
        return atlas_dic[data_len]


def _data_to_giiform(data: np.ndarray) -> np.ndarray:
    """
    Convert the data to GIFTI form(time series type are vertex*time).

    Parameters
    ----------
    data

    Returns
    -------

    """
    shape = data.shape

    if len(shape) == 1:  # as it is, 1D
        return data
    elif len(shape) == 2 and shape[0] == 1:  # CIFTI's dsclaler mode
        return data.flatten()
    elif len(shape) == 2 and shape[0] > shape[1]:  # GIFTI form: vertex*time
        return data
    elif len(shape) == 2 and shape[0] < shape[1]:  # CIFTI form: time * vertex
        return data.T


def _data_to_npform(data: np.ndarray) -> np.ndarray:
    """
    Convert the data to NUMPY form(
    * time series type are time*vertex
    * scaler/shape data are 1D array(cifti are in shape=(,vertex_number) for convinence of uniform API of cii[,n])

    Parameters
    ----------
    data

    Returns
    -------

    """

    shape = data.shape

    # density_info = pd.read_csv(__base__ / 'S1200/fslr_vertex/density_info.csv')  # change path when upload to the package
    # vertex_list = density_info['vertex_n'].to_numpy()

    if len(shape) == 1:  # as it is, 1D
        return data
    elif len(shape) == 2 and shape[0] == 1:  # CIFTI's dsclaler mode
        return data.flatten()
    elif len(shape) == 2 and shape[0] > shape[1]:  # GIFTI form: vertex*time
        return data.T
    elif len(shape) == 2 and shape[0] < shape[1]:  # time * vertex
        return data


def _data_to_ciiform(data: np.ndarray) -> np.ndarray:
    """
    Convert the data to NUMPY form(
    * time series type are time*vertex
    * scaler/shape data are 1D array(cifti are in shape=(,vertex_number) for convinence of uniform API of cii[,n])

    Parameters
    ----------
    data

    Returns
    -------

    """

    shape = data.shape

    if len(shape) == 1:  # 1D
        return np.array([data])
    elif len(shape) == 2 and shape[0] == 1:  # CIFTI's dsclaler mode
        return data
    elif len(shape) == 2 and shape[0] > shape[1]:  # GIFTI form: vertex * time
        return data.T
    elif len(shape) == 2 and shape[0] < shape[1]:  # time * vertex
        return data


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


@deprecated(version='0.1.0', reason="please use FSLR")
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
    _, _, data_len, _ = judge_density(data)
    data = _data_to_giiform(data)

    fsl_info = pd.read_csv(__base__ / "S1200/fslr_vertex/density_info.csv")
    gii2_vertexn = fsl_info.query('structure == "hmmw"')['vertex_n'].to_numpy()
    if _data_to_ciiform(data).shape[1] not in gii2_vertexn:
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


def gen_gii(data) -> tuple:
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
    density_info = pd.read_csv(__base__ / 'S1200/fslr_vertex/density_info.csv')
    density, structure, data_len, data = judge_density(data)
    gii_values = density_info.query('structure in ("L", "R", "LR", "hmmw", "gii2")')['vertex_n'].to_numpy()

    data = _data_to_giiform(data)

    if structure not in ('L', 'R', 'LR', 'hmmw', 'gii2'):
        raise ValueError("Not a legal structure.")

    if structure in ('L', 'R', 'LR'):
        data = reverse_mw(data)

    if structure in ('L', 'R', 'hmmw'):
        if structure == 'hmmw' and hm is None:
            raise ValueError('hmmw requires the assignment of hemisphere!')
        else:
            hm = structure
        return gen_gii_hm(data, hm)
    else:
        hemimw = density_info.query('density == @density and structure == "hmmw"')['vertex_n'].values[0]
        if len(data.shape) == 1:
            lh = data[:hemimw]
            rh = data[hemimw:]
        elif len(data.shape) == 2:
            lh = data[:hemimw, :]
            rh = data[:hemimw, :]

        return gen_gii_hm(lh, 'L'), gen_gii_hm(rh, 'R')


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
    if len(data.shape) >= 3:
        raise ValueError("Input data should be at most 2 dimensions!")

    if len(data.shape) == 2 and data.shape[0] == 1:
        dtype = 'scaler'
        data = data.flatten()
    elif len(data.shape) == 1:
        dtype = 'scaler'
    else:
        dtype = 'series'

    return dtype, _data_to_npform(data)


def judge_density(data: np.ndarray) -> tuple[str, str, str, np.ndarray]:
    """
    Determine the surface density and structure from the data's length.
    Remember some structures may have the same vertex number.
    fslr-4k and fslr-8k are symmetric and thus have the same vertex number for both hemispheres.

    Parameters
    ----------
    data : np.ndarray
        The input data array.

    Returns
    -------
    density : str
        The density of the surface (e.g., '32k').
    structure : str
        The structure type (e.g., 'L', 'R', 'LR').
    data_len : int
        The length of the data.
    data : np.ndarray
        The processed data.
    """

    density_info = pd.read_csv(
        __base__ / 'S1200/fslr_vertex/density_info.csv')  # change path when upload to the package
    vertex_list = density_info['vertex_n'].to_numpy()

    dtype, data = _judge_data_type(data)
    if dtype == 'series':
        data_len = data.shape[1]
    if dtype == 'scaler':
        data_len = data.shape[0]

    if data_len not in vertex_list:
        raise ValueError(f"This is not any hemisphere!")

    q_result = density_info.query('vertex_n == @data_len')
    density = q_result['density'].values[0]
    structure = q_result['structure'].values[0]

    return density, structure, data_len, data


def remove_mw(data: np.ndarray, hm:bool=None) -> np.ndarray:
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
    # Determine the density, structure, data length, and processed data
    density, structure, data_len, data = judge_density(data)
    # Convert data to CIFTI form for consistent processing
    data = _data_to_ciiform(data)
    # Load density information from CSV
    density_info = pd.read_csv(__base__ / 'S1200/fslr_vertex/density_info.csv')
    # Structures that include medial wall ('hmmw' for half hemisphere with medial wall, 'gii2' for both hemispheres)
    need_fuck_structure = ('hmmw', 'gii2')
    # Vertex numbers for structures that need medial wall removal
    need_fuck_vertexn = density_info.query('structure in @need_fuck_structure')['vertex_n'].to_numpy()

    # If data length doesn't match structures needing removal, check for errors or return unchanged
    if data_len not in need_fuck_vertexn:
        if structure in ('MW', 'MWL', 'MWR'):
            raise ValueError('Medial wall slim to nothing?')
        return data

    # Load medial wall indices for the given density
    mnw_index = np.load(__base__ / f'S1200/fslr_vertex/fslr-{density}_mw.npy')

    # Handle half hemisphere with medial wall ('hmmw')
    if structure == 'hmmw':  # only fuck half of the hemisphere
        # Require hemisphere specification for asymmetric densities
        if hm is None and density in ('32k', '164k'):
            raise ValueError(
                "Missing hemisphere spefication for the hemisphere with medial wall is symmetric across hemispheres!")
        elif hm is None and density in ('4k', '8k'):  # hm is not None or density in ('4k', '8k')
            hm = 'L'  # 4k 8k assymetry
        # Get vertex counts for left and right hemispheres
        lh_n = density_info.query('density==@density and structure=="L"')['vertex_n'].values[0]
        rh_n = density_info.query('density==@density and structure=="R"')['vertex_n'].values[0]

        # Process left hemisphere
        if hm == 'L':
            data_nmw = np.zeros((data.shape[0], lh_n))
            mnw_index = mnw_index[:lh_n]
        # Process right hemisphere
        elif hm == 'R':
            data_nmw = np.zeros((data.shape[0], rh_n))
            hmmw = density_info.query('density==@density and structure=="hmmw"')['vertex_n'].values[0]
            mnw_index = mnw_index[lh_n:] - hmmw
        # Assign non-medial wall data to new array
        data_nmw[:, :] = data.T[mnw_index].T
    # Handle both hemispheres ('gii2')
    else:  # gii2
        # Get vertex count for both hemispheres without medial wall
        LR = density_info.query('density==@density and structure=="LR"')['vertex_n'].values[0]
        data_nmw = np.zeros((data.shape[0], LR))
        # Assign non-medial wall data to new array
        data_nmw[:, :] = data.T[mnw_index].T

    # Return data with medial wall removed
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
        data_mw = np.zeros((data.shape[0], hmmw))  # may generate (1, vertex)
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


def second_smallest(data) -> float:
    """
    Find the second smallest value in an array, ignoring NaN values.
    If the array has only one non-NaN value, return the minimum value.

    Parameters
    ----------
    data : numpy.ndarray
        Input array that may contain NaN values

    Returns
    -------
    float
        The second smallest value in the array or the minimum if only one non-NaN value exists
    """
    return np.partition(data[~np.isnan(data)].flatten(), 1)[1] if len(data[~np.isnan(data)]) > 1 else np.nanmin(data)


# second largest
def second_largest(data: np.ndarray) -> float:
    """
    Find the second largest value in an array, ignoring NaN values.
    If the array has only one non-NaN value, return the maximum value.

    Parameters
    ----------
    data : numpy.ndarray
        Input array that may contain NaN values

    Returns
    -------
    float
        The second largest value in the array or the maximum if only one non-NaN value exists
    """
    return np.partition(data[~np.isnan(data)].flatten(), -2)[-2] if len(data[~np.isnan(data)]) > 1 else np.nanmax(data)


def min_max(data) -> list[float, float]:
    """
    Find the minimum and maximum values in an array, ignoring NaN values.

    Parameters
    ----------
    data : numpy.ndarray
        Input array that may contain NaN values

    Returns
    -------
    tuple
        The minimum and maximum values in the array
    """
    return np.nanmin(data), np.nanmax(data)


def get_nomw_vertex_n():
    """
    Get the vertex number of the data without medial wall.

    Returns
    -------
    : list
        List of vertex number of the data without medial wall.
    """
    density_info = pd.read_csv(__base__ / 'S1200/fslr_vertex/density_info.csv')
    return density_info.query('structure == "LR"')['vertex_n'].values