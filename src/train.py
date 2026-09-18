import numpy as np
import jax.numpy as jnp

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

from data import NUM_FEATURES, SingleCellDataset, collate_fn
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
    x = jnp.broadcast_to(x[None],   (repeat, *x.shape))
    m = bernoulli(rng0, maskrate, x.shape) & bernoulli(rng1, 1-skiprate/repeat, [repeat, 1, 1])
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

    x  = norm_fn(x)
    y0 = apply_fn(param, x, cfg, training=True, rngs=rng)
    y1 = jax.nn.one_hot(lab, y0.shape[-1])
    y2 = optax.smooth_labels(y1, 0.1)

    loss = optax.softmax_cross_entropy(y0, y2)
    loss = jnp.mean(loss)

    acc  = jnp.argmax(y0, -1) == lab
    npos = jnp.sum(acc[:, None] * msk, 0)
    nall = jnp.sum(msk, 0)
    acc  = jnp.mean(acc) * 100
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
    x  = norm_fn(x)
    y0 = state.apply_fn(state.params, x, cfg)
    y1 = jax.nn.one_hot(lab, y0.shape[-1])

    loss = optax.softmax_cross_entropy(y0, y1)
    loss = jnp.mean(loss)

    acc  = jnp.argmax(y0, -1) == lab
    acc  = jnp.mean(acc) * 100
    return loss, acc


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='MaskedPruningModel: feature selection via non-linear maksed pruning model')
    parser.add_argument('--name', type=str, default='default')
    parser.add_argument('--tgtacc', type=float, default=80)
    parser.add_argument('--numhead', type=int, default=8)
    parser.add_argument('--maskrate', type=float, default=0.15)
    parser.add_argument('--batchsize', type=int, default=8*8)
    parser.add_argument('--batchrepeat', type=int, default=8)
    parser.add_argument('--learnrate', type=float, default=1e-3)
    parser.add_argument('--weightdecay', type=float, default=1e-3)
    parser.add_argument('--decayrate', type=float, default=0.95)
    parser.add_argument('--patience', type=int, default=8*8)
    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--seed', type=int, default=20201212)
    args = parser.parse_args()
    print('#JAX:', jax.version.__version__)
    print(args)
    print()

    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng, drop_rng, sample_rng = jax.random.split(rng, 4)
    batch_rng = {'params': init_rng, 'dropout': drop_rng}

    dataset = SingleCellDataset()
    trainset, validset = dataset.split(2**12, args.batchsize*2)
    aug_fn = partial(aug_fn, repeat=args.batchrepeat, maskrate=args.maskrate, skiprate=1e-2)
    trainloader = DataLoader(trainset, batch_size=args.batchsize, shuffle=True, drop_last=True, \
                             collate_fn=collate_fn, num_workers=8)
    print('##trainloader:', len(trainset), len(trainloader), args.batchsize)
    validloader = DataLoader(validset, batch_size=args.batchsize*args.batchrepeat, shuffle=False, drop_last=False, \
                             collate_fn=collate_fn, num_workers=8)
    print('##validloader:', len(validset), len(validloader), args.batchsize*args.batchrepeat)
    #for batch in tqdm(trainloader): pass
    for batch in trainloader: break
    print()

    print('#building...')
    epochsize = len(trainloader)
    warmupsize = int(math.ceil(math.log(1e-6, 1-1e-3) // epochsize)) * epochsize
    model = MaskedPruningModel(args.numhead)
    sched = optax.warmup_exponential_decay_schedule(0, args.learnrate, warmupsize, epochsize, args.decayrate)
    early = early_stopping.EarlyStopping(min_delta=1e-3, patience=args.patience-1)
    optim = optax.lamb(learning_rate=sched, weight_decay=args.weightdecay, mask=mask_fn)
    state = init_step(*batch[:-1], batch_rng, model, optim)
    print(model.tabulate(batch_rng, *batch[:-1], depth=1))


    print('#training...')
    early = early.update(-args.tgtacc)
    feature, numfeat = np.ones(NUM_FEATURES, dtype=bool), NUM_FEATURES
    for epoch in range(9999):
        if state.step >= warmupsize and early.has_improved and numfeat > 1:
            feature[check[-max(numfeat//20, 1):]] = False
            early = early.reset().update(-args.tgtacc)
            state = state.replace(step=max(state.step - len(trainloader), warmupsize))

        trainstat, numpos, numall = [], 0, 0
        for batch in tqdm(trainloader):
            rng, drop_rng, mask_rng = jax.random.split(rng, 3)
            batch_rng = {'dropout': drop_rng, 'maskout': mask_rng}
            x, cfg, lab = batch; x = x * feature
            state, loss, acc, npos, nall = train_step(state, x, cfg, lab, batch_rng)
            trainstat.append([float(loss), float(acc)])
            numpos = numpos + np.array(npos)
            numall = numall + np.array(nall)
        trainstat = np.mean(trainstat, 0)
        check = numpos / numall.clip(1, None)
        check = np.where(feature, check, -1)
        check = np.argsort(check)

        validstat = []
        for batch in tqdm(validloader):
            x, cfg, lab = batch; x = x * feature
            loss, acc = valid_step(state, x, cfg, lab)
            validstat.append([float(loss), float(acc)])
        validstat = np.mean(validstat, 0)

        lr = sched(state.step)
        numfeat = np.sum(feature)
        early = early.update(-validstat[1])
        if early.has_improved:
            if args.save is not None:
                ckfn = args.save + '/feature%06d' % numfeat
                ckpt = {'params': tree_map(partial(prune_fn, msk=feature), state.params), 'feature': feature}
                checkpoints.save_checkpoint(ckfn, ckpt, epoch, keep=1)
            print('#epoch[%03d]:  %.4f %.2f%%  %.4f %.2f%%  %.2e %d *' % (epoch, *trainstat, *validstat, lr, numfeat))
        else:
            print('#epoch[%03d]:  %.4f %.2f%%  %.4f %.2f%%  %.2e %d'   % (epoch, *trainstat, *validstat, lr, numfeat))
        if early.should_stop: break
    print('#done!!!')

