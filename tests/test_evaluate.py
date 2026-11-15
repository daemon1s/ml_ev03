import unittest

import numpy as np

from detector.evaluate import (
    average_precision, best_threshold, collect, detections, iou_matrix, match_image,
    mean_average_precision, precision_recall,
)


def pair(boxes, scores):
    return np.asarray(boxes, dtype='float64').reshape(-1, 4), np.asarray(scores, dtype='float64')


class EvaluateTest(unittest.TestCase):
    def test_iou_of_identical_and_disjoint_boxes(self):
        overlaps = iou_matrix([[0, 0, 10, 10]], [[0, 0, 10, 10], [20, 20, 30, 30], [5, 0, 15, 10]])
        self.assertAlmostEqual(overlaps[0, 0], 1.0)
        self.assertAlmostEqual(overlaps[0, 1], 0.0)
        self.assertAlmostEqual(overlaps[0, 2], 50 / 150)

    def test_match_image_uses_each_truth_once(self):
        truth = np.asarray([[0, 0, 10, 10]], dtype='float64')
        boxes = np.asarray([[0, 0, 10, 10], [0, 0, 10, 10]], dtype='float64')
        self.assertEqual(match_image(truth, boxes, 0.5).tolist(), [True, False])

    def test_match_image_without_truth_or_predictions(self):
        self.assertEqual(match_image(np.zeros((0, 4)), np.asarray([[0, 0, 5, 5]], dtype='float64'), 0.5).tolist(), [False])
        self.assertEqual(match_image(np.asarray([[0, 0, 5, 5]], dtype='float64'), np.zeros((0, 4)), 0.5).tolist(), [])

    def test_perfect_detector_has_ap_one(self):
        truths = [np.asarray([[0, 0, 10, 10]], dtype='float64'), np.asarray([[5, 5, 20, 20]], dtype='float64')]
        detected = [pair(truths[0], [0.9]), pair(truths[1], [0.8])]
        ap50, ap = mean_average_precision(detected, truths)
        self.assertAlmostEqual(ap50, 1.0)
        self.assertAlmostEqual(ap, 1.0)

    def test_false_positive_ranked_first_lowers_ap(self):
        truths = [np.asarray([[0, 0, 10, 10]], dtype='float64')]
        detected = [pair([[50, 50, 60, 60], [0, 0, 10, 10]], [0.9, 0.5])]
        scores, flags = collect(detected, truths, 0.5)
        self.assertAlmostEqual(average_precision(scores, flags, 1), 0.5)

    def test_precision_recall_counts(self):
        truths = [np.asarray([[0, 0, 10, 10], [30, 30, 40, 40]], dtype='float64')]
        detected = [pair([[0, 0, 10, 10], [60, 60, 70, 70]], [0.9, 0.6])]
        row = precision_recall(detected, truths, 0.5)
        self.assertEqual((row['verdaderos_positivos'], row['falsos_positivos'], row['falsos_negativos']), (1, 1, 1))
        self.assertAlmostEqual(row['f1'], 0.5)

    def test_threshold_is_chosen_to_cut_low_confidence_false_positives(self):
        truths = [np.asarray([[0, 0, 10, 10]], dtype='float64')]
        detected = [pair([[0, 0, 10, 10], [60, 60, 70, 70]], [0.9, 0.3])]
        chosen = best_threshold(detected, truths)
        self.assertGreater(chosen['umbral'], 0.3)
        self.assertAlmostEqual(chosen['f1'], 1.0)

    def test_detections_caps_sorts_and_drops_masked_boxes(self):
        output = {
            'boxes': np.asarray([[[0, 0, 5, 5], [10, 10, 20, 20], [-1, -1, -1, -1], [30, 30, 40, 40]]], dtype='float64'),
            'confidence': np.asarray([[0.2, 0.9, -1.0, 0.5]]),
        }
        boxes, scores = detections(output, 320, 2)[0]
        self.assertEqual(len(boxes), 2)
        self.assertEqual(scores.tolist(), [0.9, 0.5])

    def test_detections_clips_boxes_to_the_image(self):
        output = {'boxes': np.asarray([[[-10, -10, 400, 400]]], dtype='float64'), 'confidence': np.asarray([[0.7]])}
        boxes, _ = detections(output, 320, 5)[0]
        self.assertEqual(boxes.tolist(), [[0, 0, 320, 320]])


if __name__ == '__main__':
    unittest.main()
