"""Run training on prepared folds using the current Python interpreter."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True,
                        help='Output directory from prepare_dataset_for_cv.py.')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs'))
    parser.add_argument('--folds', type=int, nargs='+', default=list(range(5)))
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate input paths and print commands without training.')
    parser.add_argument('train_args', nargs=argparse.REMAINDER,
                        help='Additional training arguments after --.')
    args = parser.parse_args()
    extra = args.train_args
    if extra[:1] == ['--']:
        extra = extra[1:]
    reserved = {'--cv_train_h5_path', '--cv_valid_h5_path',
                '--final_test_h5_path', '--save'}
    if any(arg.split('=', 1)[0] in reserved for arg in extra):
        parser.error('Dataset paths and --save are managed by this runner.')
    if any(fold < 0 for fold in args.folds) or len(set(args.folds)) != len(args.folds):
        parser.error('--folds must contain distinct nonnegative fold indices.')
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    script = Path(__file__).resolve().parent / 'model' / 'train-tissue.py'
    commands = []
    for fold in args.folds:
        fold_dir = data_dir / 'cv_folds' / f'fold_{fold}'
        paths = [fold_dir / 'train.h5', fold_dir / 'valid.h5',
                 data_dir / 'initial_split' / 'test.h5']
        for path in paths:
            if not path.is_file():
                parser.error(f'Missing prepared dataset: {path}')
        destination = output_dir / f'fold_{fold}'
        if not args.dry_run and destination.exists() and any(destination.iterdir()):
            parser.error(f'Output directory is not empty: {destination}')
        # Preserve the documented python -s isolation in the training process.
        python_flags = ['-s'] if sys.flags.no_user_site else []
        command = [sys.executable, *python_flags, '-u', str(script),
                   '--cv_train_h5_path', str(paths[0]),
                   '--cv_valid_h5_path', str(paths[1]),
                   '--final_test_h5_path', str(paths[2]),
                   '--save', str(destination / 'checkpoints'), *extra]
        commands.append((destination, command))
    env = os.environ.copy()
    env.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.20')
    env['PYTHONIOENCODING'] = 'utf-8'
    for destination, command in commands:
        print(json.dumps(command, ensure_ascii=False), flush=True)
        if args.dry_run:
            continue
        destination.mkdir(parents=True, exist_ok=True)
        (destination / 'command.json').write_text(
            json.dumps(command, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        with (destination / 'train.log').open('w', encoding='utf-8') as log:
            print(f'Training started. Follow progress in {log.name}', flush=True)
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                           env=env, check=True)
        print(f'Training completed. Results: {destination / "train.log"}', flush=True)


if __name__ == '__main__':
    main()
