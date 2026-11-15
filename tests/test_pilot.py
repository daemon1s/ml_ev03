import unittest


class PilotDataTest(unittest.TestCase):
    def test_yolo_center_coordinates_become_pixel_corners(self):
        from pilot import box_xyxy

        self.assertEqual(box_xyxy('0 0.5 0.5 0.5 0.25', 320), (80, 120, 240, 200))

    def test_out_of_frame_label_is_rejected(self):
        from pilot import box_xyxy

        with self.assertRaises(ValueError):
            box_xyxy('0 0.1 0.5 0.4 0.25', 320)
