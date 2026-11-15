import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from audit_review import Review, crop_evidence, export_train, history_rows, scan_train, verdict


class AuditReviewTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.train = self.root / 'dataset' / 'train'
        (self.train / 'images').mkdir(parents=True)
        (self.train / 'labels').mkdir()

    def example(self, name, label):
        Image.new('RGB', (100, 80), 'white').save(self.train / 'images' / f'{name}.png')
        (self.train / 'labels' / f'{name}.txt').write_text(label, encoding='utf-8')

    def test_scan_groups_identical_photos_and_tracks_invalid_labels(self):
        self.example('one', '0 .5 .5 .2 .25\n')
        self.example('two', '0 .5 .5 .2 .25\n')
        self.example('broken', '0 .05 .5 .4 .2\n')
        cases = scan_train(self.train)
        self.assertEqual(len(cases), 2)
        matching = next(case for case in cases if len(case.images) == 2)
        self.assertEqual(crop_evidence(matching.images[0], matching.boxes[0]).size, (40, 40))
        self.assertTrue(next(case for case in cases if case.errors).errors)

    def test_only_clear_patents_leave_the_review_queue(self):
        approved = verdict(json.dumps({'result': 'correcta', 'reason': 'patente_completa'}))
        self.assertEqual(approved, ('correcta', 'patente_completa'))
        with self.assertRaises(ValueError):
            verdict(json.dumps({'result': 'correcta', 'reason': 'detalle_insuficiente'}))

    def test_decisions_persist_and_export_only_changes_the_copy(self):
        self.example('one', '0 .5 .5 .2 .25\n')
        case = scan_train(self.train)[0]
        database = self.root / 'review.sqlite3'
        with Review(database) as review:
            review.save_decision(case, case.boxes[0].key, 'remove')
        with Review(database) as review:
            destination = export_train([case], review, self.root / 'export')
        original = self.train / 'labels' / 'one.txt'
        self.assertEqual(original.read_text(encoding='utf-8'), '0 .5 .5 .2 .25\n')
        self.assertEqual((destination / 'labels' / 'one.txt').read_text(encoding='utf-8'), '')

    def test_invalid_annotation_blocks_export_until_confirmed(self):
        self.example('one', 'bad\n')
        case = scan_train(self.train)[0]
        with Review(self.root / 'review.sqlite3') as review:
            with self.assertRaises(ValueError):
                export_train([case], review, self.root / 'export')
            review.save_decision(case, 'image', 'negative')
            destination = export_train([case], review, self.root / 'export')
        self.assertEqual((destination / 'labels' / 'one.txt').read_text(encoding='utf-8'), '')

    def test_history_lists_only_audited_boxes_with_human_decision(self):
        self.example('one', '0 .5 .5 .2 .25\n0 .2 .2 .1 .1\n')
        self.example('two', '0 .5 .5 .2 .25\n')
        first, second = sorted(scan_train(self.train), key=lambda case: case.images[0].name)
        with Review(self.root / 'review.sqlite3') as review:
            self.assertEqual(history_rows([first, second], review, 'modelo'), [])
            review.save_inspection(first, first.boxes[0], 'modelo', 'correcta', 'patente_completa')
            review.save_inspection(first, first.boxes[1], 'modelo', 'dudosa', 'objeto_ambiguo')
            review.save_decision(first, first.boxes[1].key, 'keep')
            rows = history_rows([first, second], review, 'modelo')
        self.assertEqual(len(rows), 2)
        self.assertEqual([(row[1], row[2], row[3], row[4]) for row in rows], [
            (first.boxes[0].key, 'correcta', 'patente_completa', None),
            (first.boxes[1].key, 'dudosa', 'objeto_ambiguo', 'keep'),
        ])
        self.assertIs(rows[0][0], first)


if __name__ == '__main__':
    unittest.main()
