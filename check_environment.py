"""Check the sAge runtime with synthetic input; no dataset is required."""

import argparse
from importlib.metadata import version
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-gpu', action='store_true')
    args = parser.parse_args()

    model_dir = Path(__file__).resolve().parent / 'model'
    sys.path.insert(0, str(model_dir))

    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax.training.early_stopping import EarlyStopping
    from data import NUM_FEATURES
    from model import MaskedPruningModel

    print('Python:', sys.version.split()[0])
    for package in ('jax', 'jaxlib', 'flax', 'optax', 'numpy', 'scipy',
                    'h5py', 'scikit-learn', 'torch', 'orbax-checkpoint'):
        print(f'{package}: {version(package)}')
    print('JAX devices:', jax.devices())
    if args.require_gpu and not any(device.platform == 'gpu' for device in jax.devices()):
        raise RuntimeError('No JAX GPU device detected. Check the CUDA backend and driver.')

    model = MaskedPruningModel(numhead=1)
    x = jnp.ones((2, NUM_FEATURES), dtype=jnp.float32)
    cfg = jnp.zeros((2, 1), dtype=jnp.int16)
    params = model.init(jax.random.PRNGKey(0), x, cfg)
    logits = np.asarray(model.apply(params, x, cfg))
    if logits.shape != (2, 6) or not np.isfinite(logits).all():
        raise RuntimeError(f'Invalid model output: {logits.shape}')
    print('Model forward pass:', logits.shape)

    training = runpy.run_path(str(model_dir / 'train-tissue.py'))
    update = training['update_early_stopping']
    state, improved = update(EarlyStopping(patience=2), 1.0)
    if not improved:
        raise RuntimeError('Early-stopping improvement check failed.')
    state, improved = update(state, 2.0)
    if improved:
        raise RuntimeError('Early-stopping non-improvement check failed.')
    print('PASS: imports, model forward pass, and early-stopping API.')


if __name__ == '__main__':
    main()
