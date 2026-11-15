import tempfile
import unittest
from pathlib import Path

from detector.train import merge_histories, plot_curves, write_history


class TrainHelpersTest(unittest.TestCase):
    def test_merge_histories_concatenates_epochs_in_order(self):
        merged = merge_histories([{'loss': [3.0, 2.0], 'val_loss': [4.0, 3.5]}, {'loss': [1.5], 'val_loss': [3.0]}])
        self.assertEqual(merged['loss'], [3.0, 2.0, 1.5])
        self.assertEqual(merged['val_loss'], [4.0, 3.5, 3.0])

    def test_write_history_creates_json_and_csv_with_one_row_per_epoch(self):
        history = {'loss': [3.0, 2.0], 'val_loss': [4.0, 3.5]}
        with tempfile.TemporaryDirectory() as temp:
            write_history(history, Path(temp))
            rows = (Path(temp) / 'history.csv').read_text(encoding='utf-8').splitlines()
            self.assertTrue((Path(temp) / 'history.json').exists())
        self.assertEqual(rows[0], 'epoch,loss,val_loss')
        self.assertEqual(len(rows), 3)

    def test_plot_curves_writes_png_for_one_and_many_epochs(self):
        with tempfile.TemporaryDirectory() as temp:
            for name, history in (('uno', {'loss': [2.0], 'val_loss': [3.0]}), ('varias', {'loss': [3.0, 2.0, 1.0], 'val_loss': [4.0, 3.0, 2.5]})):
                path = Path(temp) / f'{name}.png'
                plot_curves(history, path, boundary=2)
                self.assertGreater(path.stat().st_size, 0)


if __name__ == '__main__':
    unittest.main()
