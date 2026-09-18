# -*- coding: utf-8 -*-
# prepare_dataset_for_cv.py

import h5py
import numpy as np
import os
import argparse
from data import SingleCellDataset, BaseDataset, create_cv_folds_from_dataset, FoldDataset

def prepare_dataset(original_h5_path, output_base_dir, initial_test_size_ratio=0.2, n_cv_splits=5, random_state=42):

    print(f"--- Preparing data for {original_h5_path} ---")
    os.makedirs(output_base_dir, exist_ok=True)

    # 1. 加载原始数据集
    full_dataset = SingleCellDataset(original_h5_path)

    # 计算初始测试集大小
    initial_test_size = int(len(full_dataset) * initial_test_size_ratio)
    if initial_test_size < n_cv_splits: # 确保测试集至少能分出足够的样本
        print(f"Warning: Initial test set size ({initial_test_size}) is too small for {n_cv_splits} CV splits. Adjusting to {n_cv_splits}.")
        initial_test_size = n_cv_splits # 至少保证每个CV折叠有一个样本

    # 2. 进行第一次 train/test 分割
    initial_split_save_dir = os.path.join(output_base_dir, "initial_split")
    initial_train_set, initial_test_set = full_dataset.split(size=initial_test_size, save_dir=initial_split_save_dir)

    # 3. 对第一次分割后的训练集进行 k 折交叉验证分割
    cv_folds_output_dir = os.path.join(output_base_dir, "cv_folds")
    # 重新加载 initial_train.h5 以确保数据一致性，或者直接使用返回的 initial_train_set 对象
    # 这里我们直接使用返回的 initial_train_set 对象
    create_cv_folds_from_dataset(
        dataset=initial_train_set,
        output_base_dir=cv_folds_output_dir,
        n_splits=n_cv_splits,
        random_state=random_state
    )
    print(f"--- Data preparation complete for {original_h5_path} ---")


if __name__ == "__main__":
    # 将 description 修改为英文
    parser = argparse.ArgumentParser(description="Prepares data for an HDF5 dataset, including initial train/test split and k-fold CV split.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the original HDF5 dataset.")
    parser.add_argument('--output_dir', type=str, required=True, help="Root directory to save all split data.")
    parser.add_argument('--initial_test_size_ratio', type=float, default=0.2,
                        help="Ratio of the test set in the initial split.")
    parser.add_argument('--n_cv_splits', type=int, default=5, help="Number of folds (k) for k-fold cross-validation.")
    parser.add_argument('--random_state', type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    prepare_dataset(
        args.data_path,
        args.output_dir,
        args.initial_test_size_ratio,
        args.n_cv_splits,
        args.random_state
    )
