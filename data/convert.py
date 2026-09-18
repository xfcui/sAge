#!/usr/bin/env python3

import h5py
import numpy as np

with h5py.File('hbchen.hdf5', 'r') as f:
    data  = f['matrix']['block0_values'][()]
    print(data.shape, data.dtype, np.min(data), np.max(data))

    label = f['label']['block0_values'][()] - 1
    #for i in range(label.shape[-1]):
    #    ii = np.unique(label[:, i])
    #    for j in ii:
    #        print(i, j, np.mean(label[:, i] == j))
    for i, j in enumerate(np.sort(np.unique(label[:, 0]))):
        label[label[:, 0] == j, 0] = i
    print(label.shape, label.dtype, np.min(label, 0), np.max(label, 0))

with h5py.File('xfcui.hdf5', 'w') as f:
    f.create_dataset('data',  data=data.astype(np.float32))
    f.create_dataset('label', data=label.astype(np.int8))
