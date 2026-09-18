import h5py
import numpy as np
import jax.numpy as jnp
from torch.utils.data import Dataset


NUM_FEATURES = 22919


class BaseDataset(Dataset):
    def __init__(self, dataset, mapping):
        self.dat = dataset.dat
        self.cfg = dataset.cfg
        self.lab = dataset.lab
        self.map = mapping
        assert np.max(self.map) < len(self.lab)

    def __getitem__(self, idx):
        idx = self.map[idx]
        dat = self.dat[idx]
        cfg = self.cfg[idx]
        lab = self.lab[idx]
        return dat, cfg, lab

    def __len__(self):
        return len(self.map)

    def split(self, coverage, batchsize):
        np.random.seed(20201212)
        rndmap = self.map.copy()
        np.random.shuffle(rndmap)

        trainmap, validmap = [], []
        freq, tmpmap = {}, []
        for idx in rndmap:
            lab = self.lab[idx]
            if lab not in freq:
                freq[lab] = 1
                trainmap.append(idx)
            elif freq[lab] == 1:
                freq[lab] += 1
                validmap.append(idx)
            else:
                freq[lab] += 1
                tmpmap.append(idx)
        size = len(validmap) * coverage // batchsize * batchsize - len(validmap)
        trainmap.extend(tmpmap[size:])
        validmap.extend(tmpmap[:size])

        trainset = BaseDataset(dataset=self, mapping=trainmap)
        validset = BaseDataset(dataset=self, mapping=validmap)
        print('##split:', len(trainset), len(validset), coverage)
        return trainset, validset

class SingleCellDataset(BaseDataset):
    def __init__(self, fn='data/xfcui.hdf5'):
        print('#loading', fn, '...')
        with h5py.File(fn) as f:
            self.dat = f['data'][()].astype(np.float32)
            self.cfg = f['label'][:, 1:].astype(np.int16)
            self.lab = f['label'][:, 0].astype(np.int16)
            self.map = np.arange(len(self.lab), dtype=np.int64)
        assert self.dat.shape[-1] == NUM_FEATURES
        assert len(self.dat) == len(self.cfg) == len(self.lab)
        print('##data:', len(self.lab))


def collate_fn(batch, bound=False):
    dat = np.array([i[0] for i in batch], dtype=np.float32)
    cfg = np.array([i[1] for i in batch], dtype=np.int16)
    lab = np.array([i[2] for i in batch], dtype=np.int16)
    return dat, cfg, lab

