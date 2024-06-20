#!/usr/bin/env python


import numpy as np
import pandas as pd
import scipy.io as sio
def get_sub_list():
    return pd.read_csv("data/participants.tsv", sep="\t")['participant_id'].values

def average_uppertri(m):
    """
    Average the upper-triangle entry of a matrix.

    Input
    -----------------
    m: np.ndarray matrix, must be a square matrix and in the format of number.
        The input matrix


    Output
    ------------------
    : the average of the upper-triangle entry of matrix m
    """

    dim = m.shape

    if not isinstance(m, np.ndarray):
        raise ValueError("The input matrix is not in the format of numpr array!")
    elif len(dim) != 2:
        raise ValueError("The dimension of the input matrix is not two.")
    elif dim[0] != dim[1]:
        raise ValueError("The input matrix is not square.")

    m[np.isnan(m)] = 0
    # get the upper, leaving the main diagonal(k=1)
    upper_triangular = np.triu(m, k=1)

    # average
    return upper_triangular[np.nonzero(upper_triangular)].mean()



sub_list = get_sub_list()
netassignments = np.loadtxt('hierarchy2020/cortex_parcel_network_assignments.txt', dtype=np.int8)

def network_restrict_to_between(net, sys):
    # sub_net = np.zeros((17422, 882))
    sub_net = []
    for sys_i in np.arange(12):
        sys_loc = (sys == sys_i + 1)
        sub_net1 = net[sys_loc, :, :]
        sub_net2 = sub_net1[:, sys_loc, :]
        sub_net.append(sub_net2.reshape(sys_loc.sum()*sys_loc.sum(), 882))
    sub_net = np.concatenate(sub_net, axis=0)
    if sub_net.shape[0] != 17422:
        raise ValueError('Wrong number of within-system nodes!')
    else:
        return sub_net

# df = pd.DataFrame(columns=['participant_id', 'tv_within-sys'])

def within_system_tv(sub):
    print(sub)
    dfc = sio.loadmat(f'data/2-dfc/{sub}-dfc.mat')['dfc']  # 360, 360, 882
    within_net = network_restrict_to_within(dfc, netassignments)
    fc = np.corrcoef(within_net.T)
    tv_within = 1 - average_uppertri(fc)
    return tv_within

import multiprocess
pool = multiprocess.Pool(120)
results = []
for sub in sub_list:
    results.append(pool.apply_async(func=within_system_tv, args=(sub,)))
pool.close()
pool.join() # 阻塞

tvs = []
for res in results:
    tvs.append(res.get())

df = pd.DataFrame(columns=['participant_id', 'tv_within_sys'])
df['participant_id'] = sub_list
df['tv_within_sys'] = tvs
df.to_csv('data/4-3-scale/within_system_tv.csv', index=False)