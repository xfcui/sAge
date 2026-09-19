import numpy as np
import jax.numpy as jnp
import flax.linen as nn


DIM_HEAD = 64
DROP_RATE = 0.1


class Tau(nn.Module):
    nhead : int

    @nn.compact
    def __call__(self, x):
        tau = self.param('tau', nn.initializers.constant(0.5), [self.nhead, 1], jnp.float32)

        x = x.reshape(len(x), self.nhead, -1)
        x = x * tau
        x = x.reshape(len(x), -1)
        return x

class Scale(nn.Module):
    nhead : int

    @nn.compact
    def __call__(self, x):
        scale = self.param('scale', nn.initializers.constant(1.0), [self.nhead, 1], jnp.float32)

        x = x.reshape(len(x), self.nhead, -1)
        x = x * scale
        x = x.reshape(len(x), -1)
        return x

class Dense(nn.Module):
    dhead : int
    nhead : int = 1

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.dhead*self.nhead, [1], feature_group_count=self.nhead, use_bias=False)(x)
        return x

class Norm(nn.Module):
    dhead : int
    nhead : int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.dhead*self.nhead, use_bias=False)(x)
        x = nn.GroupNorm(self.nhead, use_scale=False, use_bias=False)(x)
        return x


class DenseBlock(nn.Module):
    nhead : int
    scale : int = 4

    @nn.compact
    def __call__(self, x, training:bool):
        x = Norm(DIM_HEAD, self.nhead)(x)
        g = Dense(DIM_HEAD*self.scale, self.nhead)(x)
        v = Dense(DIM_HEAD*self.scale, self.nhead)(x)
        x = nn.sigmoid(g) * v
        x = nn.Dropout(DROP_RATE)(x, not training)
        x = Dense(DIM_HEAD, self.nhead)(x)
        return x

class HeadBlock(nn.Module):
    nhead : int
    nclass : int = 6

    @nn.compact
    def __call__(self, x, training:bool):
        x = Tau(self.nhead)(x)
        x = Dense(self.nclass)(x)
        return x


class MaskedPruningModel(nn.Module):
    numhead : int

    @nn.compact
    def __call__(self, x, z, training:bool=False):
        if training: print('##compiling kernel for shape', x.shape, '...')
        x = DenseBlock(self.numhead)(x, training) \
          + nn.Embed(2,  DIM_HEAD*self.numhead)(z[:, 0]) \
          #+ nn.Embed(19, DIM_HEAD*self.numhead)(z[:, 1])
        x = DenseBlock(self.numhead)(x, training)
        x = HeadBlock(self.numhead)(x, training)
        return x
