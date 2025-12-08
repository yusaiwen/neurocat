"""
NeuroCAT utility module for neuroimaging data processing.

This module provides utilities for handling CIFTI and GIFTI files, atlas management,
brain model operations, data transformations, and surface density judgments.
Key functions include atlas retrieval, data conversion, medial wall handling,
and temporary file creation for neuroimaging workflows.

"""

import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import (Path, PosixPath)
from deprecated.sphinx import deprecated
import toml
import importlib.resources as pkg_resources
import os
import tempfile
from IPython.display import Image as IPyImage, SVG, display
from wand.image import Image as WandImage

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
    Create a temporary file.

    If the operating system is Linux, the file is created in `/dev/shm`, where files are stored in memory.
    On other operating systems, the default temporary directory is used.

    Parameters
    ----------
    suffix : str, optional
        Suffix of the created filename.
    prefix : str, optional
        Prefix of the created filename.

    Returns
    -------
    pathlib.Path
        Path to the temporary file. The user is responsible for deletion.
    """
    import platform

    if platform.system() == 'Linux':
        tmp_dir = '/dev/shm'
    else: # windows and macos
        tmp_dir = None

    # Create a temporary file in /dev/shm
    fd, fn = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=tmp_dir)
    os.close(fd) 

    return Path(fn)


def display_pdf(pdf):
    """
    Display a PDF file as SVG in Jupyter.

    Parameters
    ----------
    pdf : str
        Path to the PDF file.
    """
    svg_tmp = tmp_name('.svg')
    with WandImage(filename=pdf) as img:
        img.format = 'svg'
        img.save(filename=svg_tmp)
    display(SVG(filename=svg_tmp))
    Path(svg_tmp).unlink(missing_ok=True)


def display_pdf2(pdf):
    """
    Display a PDF file as PNG in Jupyter.

    Parameters
    ----------
    pdf : str
        Path to the PDF file.
    """
    png_tmp = tmp_name('.png')
    with WandImage(filename=pdf, resolution=300) as img:
        img.format = 'png'
        img.save(filename=png_tmp)
    display(IPyImage(filename=png_tmp, width=350))
    Path(png_tmp).unlink(missing_ok=True)


# Atlas
def _atlas_npar(atlas: str, par: int) -> Path:
    """
    Get the path of an atlas for a specific parcellation.

    Parameters
    ----------
    atlas : str
        Name of the atlas.
    par : int
        Parcellation number.

    Returns
    -------
    pathlib.Path
        Path to the atlas file.

    Raises
    ------
    ValueError
        If the parcellation number is not supported.
    """
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
    Get the path of an atlas by name and parcellation number.

    Parameters
    ----------
    atlas : str
        Name of the atlas.
    par : int, optional
        Parcellation number of the atlas.

    Returns
    -------
    pathlib.Path
        Absolute path to the atlas file.

    Raises
    ------
    ValueError
        If the atlas is not supported.
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
    Return the data of a GIFTI or CIFTI file.

    Parameters
    ----------
    cg : str or pathlib.Path
        Path to the CIFTI or GIFTI file.

    Returns
    -------
    numpy.ndarray
        Data from the CIFTI or GIFTI file.

    Raises
    ------
    FileNotFoundError
        If the file is not found.
    ValueError
        If the input is not a GIFTI or CIFTI object.
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


def get_surface_data(cg) -> np.ndarray:
    """
    Load surface data from CIFTI or GIFTI files.

    This function handles both single files and pairs of files (e.g., left and right hemisphere).
    When provided with a list of two files, it concatenates the data from both files.

    Parameters
    ----------
    cg : str or pathlib.Path or list
        Path to a CIFTI or GIFTI file, or a list of two file paths (left and right hemisphere).

    Returns
    -------
    numpy.ndarray
        Surface data as a NumPy array. If a list is provided, returns a concatenated array.

    Examples
    --------
    >>> # Load single hemisphere
    >>> data = get_surface_data('left_hemi.gii')

    >>> # Load both hemispheres
    >>> data = get_surface_data(['left_hemi.gii', 'right_hemi.gii'])

    Notes
    -----
    This function relies on `get_cii_gii_data` to load individual files.
    """

    if isinstance(cg, list):
        cg1 = get_cii_gii_data(cg[0])
        cg2 = get_cii_gii_data(cg[1])
        return np.concatenate([cg1, cg2])
    else:
        return get_cii_gii_data(cg)


def _get_bm_from_s1200(hm=None) -> nib.cifti2.BrainModelAxis:
    """
    Read the brain model from the S1200 sulcus file.

    Parameters
    ----------
    hm : {'L', 'R'}, optional
        Hemisphere specification.

    Returns
    -------
    nib.cifti2.BrainModelAxis
        Brain model axis object.
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


def get_bm(hm: str) -> nib.cifti2.cifti2_axes.BrainModelAxis:
    """
    Get the brain model axis for a specified hemisphere.

    Parameters
    ----------
    hm : {'L', 'R', 'LR'}
        Hemisphere specification.

    Returns
    -------
    nib.cifti2.cifti2_axes.BrainModelAxis
        Brain model axis object.
    """
    # if hm not in ['lh', 'rh', 'lhrh']: # since this method could be accessed outside. Hence, check the legality
    #     raise ValueError("Not legal value for hemisphere specification!")
    if hm in ['L', 'R']:
        return _get_bm_from_s1200(hm)
    else:  # , 'LR'
        return _get_bm_from_s1200('L') + _get_bm_from_s1200('R')


def _atlas2array(atlas):
    """
    Convert an atlas to a NumPy array.

    Parameters
    ----------
    atlas : str, tuple, list, or numpy.ndarray
        Atlas input.

    Returns
    -------
    numpy.ndarray
        Atlas as a NumPy array.

    Raises
    ------
    Exception
        If the atlas type is unknown or unsupported.
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
    Determine the atlas based on the length of the data.

    Parameters
    ----------
    data_len : int
        Length of the data.

    Returns
    -------
    str
        Name of the atlas.

    Raises
    ------
    ValueError
        If the data length is not recognized.
    """
    atlas_df = pd.read_csv(__base__ / 'atlas/atlas.csv')
    # convert this dataframe to a dictionary
    atlas_dic = dict(zip(atlas_df['n_par'], atlas_df['atlas']))
    if data_len not in atlas_dic.keys():
        raise ValueError(f"Not recognized data length:{data_len}")
    else:
        return atlas_dic[data_len]


def data_to_giiform(data: np.ndarray) -> np.ndarray:
    """
    Convert data to GIFTI form.

    Parameters
    ----------
    data : numpy.ndarray
        Input data.

    Returns
    -------
    numpy.ndarray
        Data in GIFTI form.
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


def data_to_npform(data: np.ndarray) -> np.ndarray:
    """
    Convert data to NumPy form.

    Parameters
    ----------
    data : numpy.ndarray
        Input data.

    Returns
    -------
    numpy.ndarray
        Data in NumPy form.
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


def data_to_ciiform(data: np.ndarray) -> np.ndarray:
    """
    Convert data to CIFTI form.

    Parameters
    ----------
    data : numpy.ndarray
        Input data.

    Returns
    -------
    numpy.ndarray
        Data in CIFTI form.
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
    """Concatenate left and right hemisphere files and unmask the medial wall.

    Parameters
    ----------
    lh : str or pathlib.Path
        Path to the left hemisphere file.
    rh : str or pathlib.Path
        Path to the right hemisphere file.
    den : str, optional
        Resolution string (default is '32k').
    p : pathlib.Path, optional
        Base path.

    Returns
    -------
    dict
        Dictionary with 'left' and 'right' arrays after unmasking.
    """

    l_data, r_data = nib.load(lh).agg_data(), nib.load(rh).agg_data()

    return unmask_medial(l_data, r_data, den)


def thresh_array(data, threshold):
    """
    Threshold a NumPy array.

    Parameters
    ----------
    data : numpy.ndarray
        Array to threshold.
    threshold : float
        Threshold value.

    Returns
    -------
    numpy.ndarray
        Thresholded array.
    """
    return data * (data >= threshold)


@deprecated
def unmask_medial(lh, rh, den="32k", atlas="fsLR", threshold=float('-inf')):
    """Unmask the medial wall for hemisphere data arrays.

    Parameters
    ----------
    lh : numpy.ndarray
        Left hemisphere data array.
    rh : numpy.ndarray
        Right hemisphere data array.
    den : str, optional
        Resolution (default '32k').
    atlas : str, optional
        Atlas identifier (default 'fsLR').
    threshold : float, optional
        Threshold value to apply to both hemispheres.

    Returns
    -------
    dict
        A dictionary with 'left' and 'right' arrays where the medial wall is unmasked.
    """

    lh, rh = thresh_array(lh, threshold), thresh_array(rh, threshold)

    nomedialwall_L, nomedialwall_R = cii_2_64k(FSLR['S1200_tp']['s1200_medialwall'], True)

    return dict(left=np.multiply(lh, 1 - nomedialwall_L),
                right=np.multiply(rh, 1 - nomedialwall_R))


@deprecated(version='0.1.0', reason="please use FSLR")
def _get_fslr_vertex() -> tuple:
    """Get fsLR CIFTI's 59k vertex indices.

    Returns
    -------
    tuple
        A tuple of vertex indices for 59k vertices. The first element is for the left hemisphere and the second for the right hemisphere.
    """
    lh = FSLR['vertex_len']['L']  # 29696
    dscalar_ref = __base__ / FSLR['S1200_tp']['s1200_sulc']
    bm_ref = nib.load(dscalar_ref).header.get_axis(1)
    vertex = bm_ref.vertex

    return vertex[:lh], vertex[lh:]


def con_path_list(path: os.PathLike, ls: list) -> list:
    """
    Concatenate a parent path with multiple child paths.

    Parameters
    ----------
    path : os.PathLike
        Parent directory path.
    ls : list
        List of child relative paths.

    Returns
    -------
    list
        List of concatenated paths.

    """
    path = Path(path)
    con = []
    for file in ls:
        con.append(path / file)
    return con


def gen_gii_hm(data, hm) -> nib.GiftiImage:
    """Create a GIFTI object for a hemisphere.

    Parameters
    ----------
    data : numpy.ndarray
        Data to save in GIFTI format, shape should be (32492, n).
    hm : {'L', 'R'}
        'L' for left hemisphere or 'R' for right hemisphere.

    Returns
    -------
    nib.GiftiImage
        GIFTI object for the specified hemisphere.

    Raises
    ------
    ValueError
        If the hemisphere is invalid or the data length is incorrect.
    """
    # check hemisphere
    if hm not in ("L", "R"):
        raise ValueError("Wrong input for hemisphere.")

    # check data length
    _, _, data_len, _ = judge_density(data)
    data = data_to_giiform(data)

    fsl_info = pd.read_csv(__base__ / "S1200/fslr_vertex/density_info.csv")
    gii2_vertexn = fsl_info.query('structure == "hmmw"')['vertex_n'].to_numpy()
    if data_to_ciiform(data).shape[1] not in gii2_vertexn:
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
    """Generate a GIFTI image for a single hemisphere.

    Parameters
    ----------
    data : numpy.ndarray
        Input data to convert to GIFTI.
    hm : {'L', 'R'}
        Hemisphere identifier.

    Returns
    -------
    nib.GiftiImage
        The GIFTI image for the specified hemisphere.

    Raises
    ------
    ValueError
        If the hemisphere identifier is invalid.
    """
    if hm not in ("L", "R"):
        raise ValueError("Wrong input for hemisphere.")

    gii_values = FSLR['fslr_gii_vertex'].values()
    # gii_values = np.array(list(FSLR['fslr_gii_vertex'].values())) * 2  # double for both hemisphere

    # whether data is time series data?
    dtype, data = _judge_data_type(data)
    data = data_to_giiform(data)

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
    """Save data as one or two GIFTI file objects.

    Parameters
    ----------
    data : numpy.ndarray or tuple
        Data to convert to GIFTI. Length should match expected vertex counts (e.g., 64984 including medial wall).
    hm : {'L', 'R', 'LR'}, optional
        If specified, limit the output to a single hemisphere. If omitted, returns both hemispheres.

    Returns
    -------
    tuple or nib.GiftiImage
        GIFTI object(s) that represent the input data. Returns a single GIFTI image for one hemisphere or
        a tuple of two GIFTI images for both hemispheres.

    Raises
    ------
    ValueError
        If the structure or data shape is invalid.
    """

    if isinstance(data, tuple):
        data = np.concatenate(data)

    # check data length
    density_info = pd.read_csv(__base__ / 'S1200/fslr_vertex/density_info.csv')
    density, structure, data_len, data = judge_density(data)
    gii_values = density_info.query('structure in ("L", "R", "LR", "hmmw", "gii2")')['vertex_n'].to_numpy()

    data = data_to_giiform(data)

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
    """Determine whether data represents a time series or a scalar.

    Notes
    -----
    GIFTI timeseries are stored as vertex*time, while CIFTI uses time*vertex.

    Parameters
    ----------
    data : numpy.ndarray
        Data to classify.

    Returns
    -------
    tuple
        A pair `(dtype, data)` where `dtype` is 'series' or 'scaler' and `data` is converted to NumPy form.

    Raises
    ------
    ValueError
        If the input array has more than two dimensions.
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

    return dtype, data_to_npform(data)


def judge_density(data: np.ndarray) -> tuple[str, str, str, np.ndarray]:
    """Determine the surface density and structure from the data length.

    Some structures share the same vertex count (e.g., fslr-4k and fslr-8k), so length alone may not be unique.

    Parameters
    ----------
    data : numpy.ndarray
        Input data array.

    Returns
    -------
    tuple
        A tuple `(density, structure, data_len, data)` where density and structure are strings.
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
    """Remove the medial wall values from surface data that include the medial wall.

    Parameters
    ----------
    data : numpy.ndarray
        Array containing surface data. If the medial wall is not present, the function will return the input unchanged.
    hm : {'L', 'R'}, optional
        Hemisphere specification for asymmetric vertex densities. Default is None.

    Returns
    -------
    numpy.ndarray
        Array with medial wall entries removed.
    """
    # Determine the density, structure, data length, and processed data
    density, structure, data_len, data = judge_density(data)
    # Convert data to CIFTI form for consistent processing
    data = data_to_ciiform(data)
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
    """Restore the medial wall positions for data currently stored without the medial wall.

    Parameters
    ----------
    data : numpy.ndarray
        Array of data without medial wall entries.
    hm : {'L', 'R'}, optional
        Hemisphere specification if density is asymmetric. Default is None.

    Returns
    -------
    numpy.ndarray
        Array with medial wall entries restored (NaN-filled where appropriate).
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
    data = data_to_ciiform(data)
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

    return data_to_npform(data_mw)


def second_smallest(data) -> float:
    """
    Find the second smallest value in an array, ignoring NaN values.

    Parameters
    ----------
    data : numpy.ndarray
        Input array that may contain NaN values.

    Returns
    -------
    float
        The second smallest value in the array, or the minimum if only one non-NaN value exists.
    """
    return np.partition(data[~np.isnan(data)].flatten(), 1)[1] if len(data[~np.isnan(data)]) > 1 else np.nanmin(data)


# second largest
def second_largest(data: np.ndarray) -> float:
    """Find the second largest value in an array, ignoring NaNs.

    Parameters
    ----------
    data : numpy.ndarray
        Input array that may contain NaN values.

    Returns
    -------
    float
        The second largest value, or the maximum if only one non-NaN value exists.
    """
    return np.partition(data[~np.isnan(data)].flatten(), -2)[-2] if len(data[~np.isnan(data)]) > 1 else np.nanmax(data)


def min_max(data) -> list[float, float]:
    """Find the minimum and maximum values in an array, ignoring NaNs.

    Parameters
    ----------
    data : numpy.ndarray
        Input array that may contain NaN values.

    Returns
    -------
    tuple
        A tuple with the minimum and maximum values in the array.
    """
    return np.nanmin(data), np.nanmax(data)


def get_nomw_vertex_n():
    """Get the vertex numbers for data without the medial wall.

    Returns
    -------
    numpy.ndarray
        Array of vertex counts for structures that exclude the medial wall (structure == 'LR').
    """
    density_info = pd.read_csv(__base__ / 'S1200/fslr_vertex/density_info.csv')
    return density_info.query('structure == "LR"')['vertex_n'].values


def cleanup_files(file_list):
    """
    Remove files from the filesystem.

    Parameters
    ----------
    file_list : list
        List of file paths to remove.
    """
    for f in file_list:
        Path(f).unlink(missing_ok=True)