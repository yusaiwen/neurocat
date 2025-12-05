from typing import Union
from pathlib import Path
import toml
import importlib.resources as pkg_resources


_base = pkg_resources.files("neurocat")  # connot import from utli.py, nor circular import
with open(_base / 'atlas.toml', 'r') as f:
    ATLAS = toml.load(f).get('atlas')


def _atlas_npar(atlas: str, par: int) -> Path:
    atlas_dic = dict(
        schaefer2018 = _base / f"atlas/Schaefer2018/Schaefer2018_{par}Parcels_7Networks_order.dlabel.nii",
        yeo2011 = _base / f"atlas/Yeo2011_{par}Networks_N1000.dlabel.nii"
    )

    file = Path(atlas_dic[atlas])

    if file.exists():
        return file
    else:
        raise ValueError(f"Parcellation number of {par} is not supported for {atlas}.")


def get_atlas(atlas: str = None, par: int = None) -> Path:
    if atlas not in ATLAS['ATLAS']:
        raise ValueError('Not supported atlas or you incorrectly type.')

    file = Path(_base / ATLAS[atlas])
    if file.is_file():
        return file.absolute()
    else:  # directory
        return _atlas_npar(atlas, par)
