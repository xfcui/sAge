# -*- coding: utf-8 -*-

import numpy as np
import jax.numpy as jnp
import os

import math
import argparse
from tqdm import tqdm
from functools import partial

import jax
import optax
import flax.linen as nn
from jax.random import bernoulli
from torch.utils.data import DataLoader
from jax.tree_util import tree_map_with_path, tree_map
from flax.training import train_state, checkpoints, early_stopping

# Import FoldDataset for loading pre-split data
from data import NUM_FEATURES, FoldDataset, collate_fn
from model import MaskedPruningModel

def mask_fn(param):
    def worker(n, p):
        return n[-1].key not in ['embedding', 'tau', 'scale', 'bias']

    return tree_map_with_path(worker, param)


def clip_fn(n, p):
    if n[-1].key in ['tau']:
        return jnp.clip(p, 0, None)
    elif n[-1].key in ['scale']:
        return jnp.clip(p, 0, 1)
    else:
        return p


def prune_fn(p, msk):
    if len(p) == len(msk):
        return p[msk]
    else:
        return p


def aug_fn(x, cfg, lab, rng, repeat, maskrate, skiprate):
    rng0, rng1 = jax.random.split(rng, 2)
    x = jnp.broadcast_to(x[None], (repeat, *x.shape))
    m = bernoulli(rng0, maskrate, x.shape) & bernoulli(rng1, 1 - skiprate / repeat, [repeat, 1, 1])
    x = jnp.where(m, 0, x)
    x = x.reshape(-1, *x.shape[2:])
    m = m.reshape(-1, *m.shape[2:])

    cfg = jnp.broadcast_to(cfg[None], (repeat, *cfg.shape))
    cfg = cfg.reshape(-1, *cfg.shape[2:])

    lab = jnp.broadcast_to(lab[None], (repeat, *lab.shape))
    lab = lab.reshape(-1, *lab.shape[2:])
    return x, cfg, lab, m


def norm_fn(x):
    x = x / jnp.sum(x, -1, keepdims=True).clip(1, None)
    x = jnp.log2(x * 1023 + 1)
    return x


def loss_fn(param, apply_fn, x, cfg, lab, rng):
    x, cfg, lab, msk = aug_fn(x, cfg, lab, rng['maskout'])

    x = norm_fn(x)
    y0 = apply_fn(param, x, cfg, training=True, rngs=rng)
    y1 = jax.nn.one_hot(lab, y0.shape[-1])
    y2 = optax.smooth_labels(y1, 0.1)

    loss = optax.softmax_cross_entropy(y0, y2)
    loss = jnp.mean(loss)

    acc = jnp.argmax(y0, -1) == lab
    npos = jnp.sum(acc[:, None] * msk, 0)
    nall = jnp.sum(msk, 0)
    acc = jnp.mean(acc) * 100
    return loss, (acc, npos, nall)


def init_step(x, cfg, rng, model, optim):
    param = model.init(rng, x, cfg)
    state = train_state.TrainState.create(apply_fn=model.apply, params=param, tx=optim)
    return state


@jax.jit
def train_step(state, x, cfg, lab, rng):
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (acc, npos, nall)), grad = grad_fn(state.params, state.apply_fn, x, cfg, lab, rng)
    state = state.apply_gradients(grads=grad)
    state = state.replace(params=tree_map_with_path(clip_fn, state.params))
    return state, loss, acc, npos, nall


@jax.jit
def valid_step(state, x, cfg, lab):
    x = norm_fn(x)
    y0 = state.apply_fn(state.params, x, cfg)
    y1 = jax.nn.one_hot(lab, y0.shape[-1])

    loss = optax.softmax_cross_entropy(y0, y1)
    loss = jnp.mean(loss)

    acc = jnp.argmax(y0, -1) == lab
    acc = jnp.mean(acc) * 100
    return loss, acc


def update_early_stopping(early_stopper_instance, metric_value):
    """Support both the historical and current Flax return conventions."""
    updated = early_stopper_instance.update(metric_value)
    if isinstance(updated, tuple):
        has_improved, state = updated
        return state, has_improved
    return updated, updated.has_improved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='sAge: feature selection via a nonlinear masked-pruning model')
    parser.add_argument('--name', type=str, default='default')
    parser.add_argument('--numhead', type=int, default=8)
    parser.add_argument('--maskrate', type=float, default=0.15)
    parser.add_argument('--batchsize', type=int, default=8 * 8)
    parser.add_argument('--batchrepeat', type=int, default=8)
    parser.add_argument('--learnrate', type=float, default=1e-3)
    parser.add_argument('--weightdecay', type=float, default=1e-3)
    parser.add_argument('--decayrate', type=float, default=0.95)
    parser.add_argument('--patience', type=int, default=8 * 8)
    parser.add_argument('--save', type=str, default=None, help="Directory to save model checkpoints and features.")
    parser.add_argument('--seed', type=int, default=20201212)
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of PyTorch DataLoader workers (default: 0 for portability).')
    parser.add_argument('--max_epochs', type=int, default=9999,
                        help='Maximum training epochs; early stopping may finish sooner.')
    # New parameters for K-fold cross-validation
    parser.add_argument('--cv_train_h5_path', type=str, required=True, help="HDF5 file path for the training set of the current CV fold.")
    parser.add_argument('--cv_valid_h5_path', type=str, required=True, help="HDF5 file path for the validation set of the current CV fold.")
    parser.add_argument('--final_test_h5_path', type=str, required=True, help="HDF5 file path for the final test set obtained from the initial split.")
    args = parser.parse_args()
    print('#JAX:', jax.version.__version__)
    for i, d in enumerate(jax.local_devices()): print('#device[%d]:' % i, d.device_kind)
    print(args)
    print()

    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng, drop_rng, sample_rng = jax.random.split(rng, 4)
    batch_rng = {'params': init_rng, 'dropout': drop_rng}

    # --- Dataset Loading ---
    # Load training and validation sets for the current CV fold
    trainset = FoldDataset(args.cv_train_h5_path)
    validset = FoldDataset(args.cv_valid_h5_path)
    # Load the final test set (not used for early stopping, only for final evaluation)
    final_testset = FoldDataset(args.final_test_h5_path)
    print(f"Loaded final test set (size: {len(final_testset)}) from {args.final_test_h5_path}")
    # --- End Dataset Loading ---

    aug_fn = partial(aug_fn, repeat=args.batchrepeat, maskrate=args.maskrate, skiprate=1e-2)
    if args.batchsize > len(trainset):
        print(f"Warning: batch_size ({args.batchsize}) is greater than training set size ({len(trainset)}). Adjusting batch_size to training set size.")
        args.batchsize = len(trainset) # Adjust batch_size if it's too large
    if args.batchsize == 0: # Prevent division by zero if training set is empty
        raise ValueError("Training set is empty, cannot proceed with training.")

    trainloader = DataLoader(trainset, batch_size=args.batchsize, shuffle=True, drop_last=True, \
                             collate_fn=collate_fn, num_workers=args.num_workers)
    print('##trainloader:', len(trainset), len(trainloader), args.batchsize)

    valid_batch_size = args.batchsize * args.batchrepeat
    if valid_batch_size > len(validset) and len(validset) > 0:
        print(f"Warning: valid_batch_size ({valid_batch_size}) is greater than validation set size ({len(validset)}). Adjusting valid_batch_size to validation set size.")
        valid_batch_size = len(validset)
    if len(validset) == 0:
        print("Warning: Validation set is empty.")
        validloader = [] # Empty list to avoid issues
    else:
        validloader = DataLoader(validset, batch_size=valid_batch_size, shuffle=False, drop_last=False, \
                                 collate_fn=collate_fn, num_workers=args.num_workers)
    print('##validloader:', len(validset), len(validloader), valid_batch_size)

    # DataLoader for the final test set
    final_test_batch_size = valid_batch_size # Can use the same batch_size as validation
    if len(final_testset) == 0:
        print("Warning: Final test set is empty.")
        final_testloader = []
    else:
        final_testloader = DataLoader(final_testset, batch_size=final_test_batch_size, shuffle=False, drop_last=False, \
                                      collate_fn=collate_fn, num_workers=args.num_workers)
    print('##final_testloader:', len(final_testset), len(final_testloader), final_test_batch_size)


    if len(trainloader) == 0:
        print("Trainloader is empty, skipping training for this fold.")
        exit()

    for batch in trainloader: break
    print()

    print('#building...')

    epochsize = len(trainloader)
    warmupsize = int(math.ceil(math.log(1e-4, 0.995) // epochsize)) * epochsize
    model = MaskedPruningModel(args.numhead)
    sched = optax.warmup_exponential_decay_schedule(0, args.learnrate, warmupsize, epochsize, args.decayrate)
    early = early_stopping.EarlyStopping(min_delta=1e-3, patience=args.patience - 1)

    optim = optax.lamb(learning_rate=sched, weight_decay=args.weightdecay, mask=mask_fn)
    state = init_step(*batch[:-1], batch_rng, model, optim)
    print(model.tabulate(batch_rng, *batch[:-1], depth=1))

    print('#training...')
    divider, minacc = 20, 80
    early, has_improved = update_early_stopping(early, -minacc)

    feature, numfeat = np.ones(NUM_FEATURES, dtype=bool), NUM_FEATURES
    for epoch in range(args.max_epochs):
        if epoch > 4 and has_improved and numfeat > 1:
            feature[check[-max(numfeat // divider, 1):]] = False
            early = early.reset()
            early, has_improved = update_early_stopping(early, -minacc)
            state = state.replace(step=max(state.step - len(trainloader), warmupsize))

        trainstat, numpos, numall = [], 0, 0
        for batch in tqdm(trainloader):
            rng, drop_rng, mask_rng = jax.random.split(rng, 3)
            batch_rng = {'dropout': drop_rng, 'maskout': mask_rng}
            x, cfg, lab = batch;
            x = x * feature
            state, loss, acc, npos, nall = train_step(state, x, cfg, lab, batch_rng)
            trainstat.append([float(loss), float(acc)])
            numpos = numpos + np.array(npos)
            numall = numall + np.array(nall)
        trainstat = np.mean(trainstat, 0)

        check = numpos / numall.clip(1, None)
        check = np.where(feature, check, -1)
        check = np.argsort(check)

        validstat = []
        if len(validloader) > 0:
            for batch in tqdm(validloader):
                x, cfg, lab = batch;
                x = x * feature
                loss, acc = valid_step(state, x, cfg, lab)
                validstat.append([float(loss), float(acc)])
            validstat = np.mean(validstat, 0)
        else:
            validstat = [float('inf'), 0.0]

        lr = sched(state.step)
        numfeat = np.sum(feature)

        early, has_improved = update_early_stopping(early, -validstat[1])

        if args.save is not None:
            current_save_dir = os.path.abspath(args.save)
        else:
            current_save_dir = None

        if has_improved:
            if current_save_dir is not None:
                ckfn = os.path.join(current_save_dir, 'feature%06d' % numfeat)
                os.makedirs(ckfn, exist_ok=True)
                np.savetxt(os.path.join(ckfn, 'feature.txt'), feature, delimiter=',', fmt='%.6f')
                ckpt = {'params': tree_map(partial(prune_fn, msk=feature), state.params), 'feature': feature}
                checkpoints.save_checkpoint(ckfn, ckpt, epoch, keep=1, overwrite=True)
            print('#epoch[%03d]:  %.4f %.2f%%  %.4f %.2f%%  %.2e %d *' % (epoch, *trainstat, *validstat, lr, numfeat))
        else:
            print('#epoch[%03d]:  %.4f %.2f%%  %.4f %.2f%%  %.2e %d' % (epoch, *trainstat, *validstat, lr, numfeat))

        if early.should_stop:
            print("#Early stopping triggered at epoch", epoch)
            break
    print('#done!!!')

    # --- Final Test Set Evaluation (Optional) ---
    # Here you can add evaluation on the final_testset to get the model's performance on unseen data
    final_test_stat = []
    if len(final_testloader) > 0:
        print("\n#Evaluating on final test set...")
        for batch in tqdm(final_testloader):
            x, cfg, lab = batch
            x = x * feature # Use the trained feature subset
            loss, acc = valid_step(state, x, cfg, lab) # Use valid_step for evaluation
            final_test_stat.append([float(loss), float(acc)])
        final_test_stat = np.mean(final_test_stat, 0)
        print(f"#Final Test: Loss = {final_test_stat[0]:.4f}, Acc = {final_test_stat[1]:.2f}%")
    else:
        print("#No final test set evaluation (empty loader).")
    # --- End Final Test Set Evaluation ---
