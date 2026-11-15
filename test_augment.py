import unittest

import numpy as np

SIZE = 320


def gradient_image():
    column = np.linspace(0, 255, SIZE, dtype='float32')
    plane = np.tile(column, (SIZE, 1))
    return np.stack([plane, plane, plane], axis=-1)


class AffineTest(unittest.TestCase):
    def test_identity_keeps_image_and_boxes(self):
        from augment import affine

        image = gradient_image()
        boxes = [(80.0, 120.0, 240.0, 200.0)]
        moved_image, moved_boxes = affine(image, boxes, 1.0, 0.0, 0.0)

        np.testing.assert_allclose(moved_image, image, atol=1.0)
        np.testing.assert_allclose(moved_boxes, boxes, atol=1e-3)

    def test_shift_moves_boxes_and_pixels_together(self):
        from augment import affine

        image = gradient_image()
        boxes = [(80.0, 120.0, 160.0, 200.0)]
        moved_image, moved_boxes = affine(image, boxes, 1.0, 20.0, 0.0)

        np.testing.assert_allclose(moved_boxes, [(100.0, 120.0, 180.0, 200.0)], atol=1e-3)
        np.testing.assert_allclose(moved_image[100, 120 + 20], image[100, 120], atol=1.5)

    def test_scale_grows_boxes_around_the_center(self):
        from augment import affine

        boxes = [(120.0, 120.0, 200.0, 200.0)]
        _, moved_boxes = affine(gradient_image(), boxes, 2.0, 0.0, 0.0)

        np.testing.assert_allclose(moved_boxes, [(80.0, 80.0, 240.0, 240.0)], atol=1e-3)

    def test_box_that_leaves_the_frame_is_dropped(self):
        from augment import affine

        boxes = [(10.0, 100.0, 60.0, 140.0), (200.0, 100.0, 250.0, 140.0)]
        _, moved_boxes = affine(gradient_image(), boxes, 1.0, -SIZE / 2, 0.0)

        self.assertEqual(len(moved_boxes), 1)
        np.testing.assert_allclose(moved_boxes[0], (40.0, 100.0, 90.0, 140.0), atol=1e-3)

    def test_mostly_cropped_box_is_dropped(self):
        from augment import affine

        boxes = [(10.0, 100.0, 110.0, 140.0)]
        _, moved_boxes = affine(gradient_image(), boxes, 1.0, -80.0, 0.0)

        self.assertEqual(len(moved_boxes), 0)


class AugmentTest(unittest.TestCase):
    def test_same_seed_gives_same_result(self):
        from augment import augment

        image = gradient_image()
        boxes = [(80.0, 120.0, 240.0, 200.0)]
        first = augment(image, boxes, np.random.default_rng(7), SIZE)
        second = augment(image, boxes, np.random.default_rng(7), SIZE)

        np.testing.assert_array_equal(first[0], second[0])
        self.assertEqual(first[1], second[1])

    def test_output_keeps_shape_range_and_valid_boxes(self):
        from augment import augment

        rng = np.random.default_rng(0)
        image = gradient_image()
        boxes = [(80.0, 120.0, 240.0, 200.0)]
        for _ in range(200):
            out_image, out_boxes = augment(image, boxes, rng, SIZE)
            self.assertEqual(out_image.shape, (SIZE, SIZE, 3))
            self.assertEqual(out_image.dtype, np.float32)
            self.assertGreaterEqual(out_image.min(), 0.0)
            self.assertLessEqual(out_image.max(), 255.0)
            for left, top, right, bottom in out_boxes:
                self.assertTrue(0 <= left < right <= SIZE)
                self.assertTrue(0 <= top < bottom <= SIZE)

    def test_never_mirrors_the_image(self):
        from augment import augment

        rng = np.random.default_rng(1)
        image = gradient_image()
        boxes = [(40.0, 120.0, 120.0, 200.0)]
        for _ in range(200):
            out_image, out_boxes = augment(image, boxes, rng, SIZE)
            row = out_image[SIZE // 2].mean(axis=-1)
            self.assertGreater(row[-40:].mean(), row[:40].mean())

    def test_inputs_are_not_modified(self):
        from augment import augment

        image = gradient_image()
        copy = image.copy()
        boxes = [(80.0, 120.0, 240.0, 200.0)]
        augment(image, boxes, np.random.default_rng(3), SIZE)

        np.testing.assert_array_equal(image, copy)
        self.assertEqual(boxes, [(80.0, 120.0, 240.0, 200.0)])


if __name__ == '__main__':
    unittest.main()
