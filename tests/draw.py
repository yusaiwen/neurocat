#!/usr/bin/env python

import numpy as np
import scipy.io as sio
from neurocat.util import atlas_2_wholebrain as a2w
from neurocat.plotting import ds


indvar_mat = sio.loadmat("indvar_6gp.mat")["indvar"]
for group in np.arange(6):
    indvar = indvar_mat[:, group]
    layer = a2w(indvar)
    ds(layer,'RdBu_r', (0, 2.7), f"gp-{group}.png", "column", outline=True)