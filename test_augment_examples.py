import unittest

import numpy as np

from augment_examples import photometric_only


class PhotometricOnlyTest(unittest.TestCase):
    def setUp(self):
        self.image = np.full((16, 16, 3), 100.0, dtype='float32')
        self.rng = np.random.default_rng(0)

    def test_default_leaves_the_image_unchanged(self):
        np.testing.assert_allclose(photometric_only(self.image), self.image)

    def test_brightness_scales_the_mean(self):
        self.assertAlmostEqual(float(photometric_only(self.image, brightness=1.3).mean()), 130.0, places=3)

    def test_noise_changes_pixels_and_stays_in_range(self):
        noisy = photometric_only(self.image, noise=10.0, rng=self.rng)
        self.assertGreater(float(noisy.std()), 0.0)
        self.assertTrue(((noisy >= 0) & (noisy <= 255)).all())


if __name__ == '__main__':
    unittest.main()
