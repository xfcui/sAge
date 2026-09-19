import h5py
import numpy as np
from torch.utils.data import Dataset
import os
from sklearn.model_selection import StratifiedKFold

NUM_FEATURES = 22919
#NUM_FEATURES = 35528

class BaseDataset(Dataset):
    def __init__(self, dat, cfg, lab):
        self.dat = dat
        self.cfg = cfg
        self.lab = lab
        self.map = np.arange(len(self.lab), dtype=np.int64)

    def __getitem__(self, idx):
        dat = self.dat[idx].copy()
        cfg = self.cfg[idx].copy()
        lab = self.lab[idx].copy()
        return dat.astype(np.float32), cfg.astype(np.int16), lab.astype(np.int16)

    def __len__(self):
        return len(self.map)

    def split(self, size, save_dir=None):

        np.random.seed(20201212)
        rndmap = self.map.copy()
        np.random.shuffle(rndmap)
        trainmap, testmap = [], []
        # Fix: Initialize freq and tmpmap
        freq = {}
        tmpmap = []
        for idx in rndmap:
            lab = self.lab[idx]
            if lab not in freq:
                freq[lab] = 1
                trainmap.append(idx)
            elif freq[lab] == 1:
                freq[lab] += 1
                testmap.append(idx)
            else:
                freq[lab] += 1
                tmpmap.append(idx)


        num_to_move = max(0, size - len(testmap))
        testmap.extend(tmpmap[:num_to_move])
        trainmap.extend(tmpmap[num_to_move:])


        trainmap = np.array(trainmap, dtype=np.int64)
        testmap = np.array(testmap, dtype=np.int64)

        trainset = BaseDataset(dat=self.dat[trainmap], cfg=self.cfg[trainmap], lab=self.lab[trainmap])
        testset = BaseDataset(dat=self.dat[testmap], cfg=self.cfg[testmap], lab=self.lab[testmap])

        print(f'##Initial split: Train size = {len(trainset)}, Test size = {len(testset)}')

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            with h5py.File(f"{save_dir}/train.h5", 'w') as f:
                f.create_dataset('data', data=self.dat[trainmap])
                f.create_dataset('label', data=np.column_stack([self.lab[trainmap], self.cfg[trainmap]]))
            with h5py.File(f"{save_dir}/test.h5", 'w') as f:
                f.create_dataset('data', data=self.dat[testmap])
                f.create_dataset('label', data=np.column_stack([self.lab[testmap], self.cfg[testmap]]))
            print(f"Saved initial train/test sets to {save_dir}/[train|test].h5")
        return trainset, testset


class SingleCellDataset(BaseDataset):

    def __init__(self, fn):
        print('#loading', fn, '...')
        with h5py.File(fn, 'r') as f:
            dat = f['data'][()]
            cfg = f['label'][:, 1:]
            lab = f['label'][:, 0]
            print(f"Data shape: {dat.shape}, Label shape: {lab.shape}")
        super().__init__(dat=dat, cfg=cfg, lab=lab)
        assert self.dat.shape[-1] == NUM_FEATURES
        assert len(self.dat) == len(self.cfg) == len(self.lab)
        print('##data:', len(self.lab))


class FoldDataset(BaseDataset):

    def __init__(self, h5_file_path):
        # print(f"#loading fold data from {h5_file_path}...")
        with h5py.File(h5_file_path, 'r') as f:
            dat = f['data'][()]
            lab_and_cfg = f['label'][()]
            lab = lab_and_cfg[:, 0]
            cfg = lab_and_cfg[:, 1:]
        super().__init__(dat=dat, cfg=cfg, lab=lab)
        # print(f"Loaded data shape: {self.dat.shape}, Label shape: {self.lab.shape} from {h5_file_path}")


def create_cv_folds_from_dataset(dataset: BaseDataset, output_base_dir: str, n_splits=5, random_state=42):

    print(f"Creating {n_splits}-fold CV splits from dataset (size: {len(dataset)}) into {output_base_dir}...")

    all_indices = np.arange(len(dataset.lab))


    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for k, (train_indices, valid_indices) in enumerate(skf.split(all_indices, dataset.lab)):
        fold_dir = os.path.join(output_base_dir, f"fold_{k}")
        os.makedirs(fold_dir, exist_ok=True)


        with h5py.File(os.path.join(fold_dir, "train.h5"), 'w') as f:
            f.create_dataset('data', data=dataset.dat[train_indices])
            f.create_dataset('label', data=np.column_stack([dataset.lab[train_indices], dataset.cfg[train_indices]]))


        with h5py.File(os.path.join(fold_dir, "valid.h5"), 'w') as f:
            f.create_dataset('data', data=dataset.dat[valid_indices])
            f.create_dataset('label', data=np.column_stack([dataset.lab[valid_indices], dataset.cfg[valid_indices]]))

        print(f"  CV Fold {k}: Train size = {len(train_indices)}, Valid size = {len(valid_indices)}")

    print(f"Finished creating {n_splits} CV folds in {output_base_dir}")


def collate_fn(batch, bound=False):
    dat = np.array([i[0] for i in batch], dtype=np.float32)
    cfg = np.array([i[1] for i in batch], dtype=np.int16)
    lab = np.array([i[2] for i in batch], dtype=np.int16)
    return dat, cfg, lab
