import unittest


class MergeManualTest(unittest.TestCase):
    def test_coco_box_becomes_normalized_yolo_center(self):
        from detector.merge_manual import coco_to_yolo

        self.assertEqual(coco_to_yolo([160, 120, 320, 240], 640, 480), '0 0.500000 0.500000 0.500000 0.500000')

    def test_box_outside_image_is_rejected(self):
        from detector.merge_manual import coco_to_yolo

        with self.assertRaises(ValueError):
            coco_to_yolo([600, 10, 100, 20], 640, 480)

    def test_labels_are_grouped_by_image(self):
        from detector.merge_manual import labels_by_image

        coco = {
            'images': [
                {'id': 1, 'file_name': 'img_001.jpg', 'width': 640, 'height': 480},
                {'id': 2, 'file_name': 'img_002.jpg', 'width': 640, 'height': 480},
            ],
            'annotations': [
                {'image_id': 1, 'bbox': [0, 0, 64, 48]},
                {'image_id': 1, 'bbox': [64, 48, 64, 48]},
                {'image_id': 2, 'bbox': [320, 240, 64, 48]},
            ],
        }
        result = labels_by_image(coco)
        self.assertEqual(len(result['img_001.jpg']), 2)
        self.assertEqual(len(result['img_002.jpg']), 1)


if __name__ == '__main__':
    unittest.main()
