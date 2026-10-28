"""Checks for positional mapping, genuine RSS minimization and split isolation."""

from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model'))
from gene_selection import (collect_masks, load_data, load_genes, read_mask,
                            regression_metrics, segmented_knee)


class GeneSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_header_preserves_case_order_and_duplicate_positions(self):
        header = self.root / 'header.txt'
        header.write_text('Zbtb1\nAbc\nZbtb1\n', encoding='utf-8')
        genes = load_genes(header)
        self.assertEqual(genes, ['Zbtb1', 'Abc', 'Zbtb1'])
        mask_path = self.root / 'feature.txt'
        mask_path.write_text('0\n0\n1\n', encoding='utf-8')
        indices = np.flatnonzero(read_mask(mask_path, 3))
        self.assertEqual(indices.tolist(), [2])
        self.assertEqual([genes[i] for i in indices], ['Zbtb1'])

    def test_header_rejects_blank_line(self):
        header = self.root / 'header.txt'
        header.write_text('A\n\nB\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            load_genes(header)

    def test_invalid_masks_fail(self):
        mask_path = self.root / 'feature.txt'
        for content in ('1\n0\n', '1\n0.5\n0\n', '1\nnan\n0\n', '0\n0\n0\n'):
            mask_path.write_text(content, encoding='utf-8')
            with self.subTest(content=content), self.assertRaises(ValueError):
                read_mask(mask_path, 3)

    def test_short_pruning_run_is_not_a_final_gene_list(self):
        folder = self.root / 'feature000003'
        folder.mkdir()
        (folder / 'feature.txt').write_text('1\n1\n1\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'at least 6'):
            collect_masks(self.root, 3, 1011)

    def test_count_mismatch_fails(self):
        folder = self.root / 'feature000003'
        folder.mkdir()
        (folder / 'feature.txt').write_text('1\n0\n1\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            collect_masks(self.root, 3, None)

    def test_knee_recovers_known_piecewise_breakpoint(self):
        # Perfect two-line curve: slopes -2 and -1 meet at 4 genes.
        x = np.arange(1, 10)
        y = np.where(x <= 4, 16 - 2 * x, 8 - (x - 4))
        rows = [{'Gene_Count': int(a), 'MAE': float(b)} for a, b in zip(x, y)]
        result = segmented_knee(rows[::-1])
        self.assertEqual(result['Gene_Count'], 4)
        self.assertLess(result['Segmented_RSS'], 1e-20)

    def test_flat_or_short_curves_fail(self):
        with self.assertRaisesRegex(ValueError, 'flat'):
            segmented_knee([{'Gene_Count': i, 'MAE': 2.0} for i in range(1, 9)])
        with self.assertRaisesRegex(ValueError, 'six'):
            segmented_knee([{'Gene_Count': i, 'MAE': float(i)} for i in range(1, 5)])

    def test_zero_mad_does_not_drop_every_point(self):
        result = segmented_knee([{'Gene_Count': i, 'MAE': value} for i, value in
                                 enumerate([8., 6., 4., 2., 2., 2., 2., 2., 2.], 1)])
        self.assertGreater(result['Gene_Count'], 0)
        self.assertEqual(result['MAD_Excluded_Counts'], [])

    def test_hdf5_shape_and_age_mapping_are_checked(self):
        path = self.root / 'train.h5'
        with h5py.File(path, 'w') as handle:
            handle['data'] = np.ones((4, 3))
            handle['label'] = np.array([[1, 1], [4, 1], [1, 1], [4, 1]])
        x, ages = load_data(path, 3, {1: 3, 4: 24})
        self.assertEqual(x.shape, (4, 3))
        self.assertEqual(ages.tolist(), [3, 24, 3, 24])
        with self.assertRaisesRegex(ValueError, 'columns'):
            load_data(path, 4, {1: 3, 4: 24})
        with self.assertRaisesRegex(ValueError, 'missing class'):
            load_data(path, 3, {1: 3})

    def test_regression_matches_for_c_and_fortran_input_layouts(self):
        rng = np.random.default_rng(33)
        x = rng.normal(size=(90, 8)).astype(np.float32)
        x[:, 7] = x[:, 0] + 1e-5 * x[:, 1]
        ages = np.where(x[:, 0] > 0, 24.0, 3.0)
        c_metrics = regression_metrics(np.ascontiguousarray(x[:60]), ages[:60],
                                       np.ascontiguousarray(x[60:]), ages[60:])
        f_metrics = regression_metrics(np.asfortranarray(x[:60]), ages[:60],
                                       np.asfortranarray(x[60:]), ages[60:])
        for name in ('PCC', 'MAE', 'RMSE'):
            self.assertAlmostEqual(c_metrics[name], f_metrics[name], places=7)


if __name__ == '__main__':
    unittest.main()
