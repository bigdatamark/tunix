# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# [MODULE DOC]
# Unit tests for tunix.processors.image_processor.ImageProcessor. The processor resizes images to the
# configured height/width and normalizes by (pixel - mean) / std, so an all-black image becomes all
# -1.0 with mean and std 127.5; a None image becomes an all-zero image. Calling the processor on a
# batch returns, per batch element, a list of processed images padded with zero images to the max
# number of images per element. Inputs accepted: numpy arrays, file paths, None, a single image, or
# lists (optionally nested). Side effects: temporary PNG files written and removed via cleanup.

import dataclasses
import os
import shutil
import tempfile
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from PIL import Image
from tunix.processors import image_processor


@dataclasses.dataclass(slots=True, kw_only=True)
class DummyConfig:
  # [CLASS DOC]
  # HELPER: minimal keyword-only, slotted dataclass config for ImageProcessor with image_height and
  # image_width (default 32), image_channels (default 3), and per-channel image_mean and image_std
  # (default 127.5 each).

  image_height: int = 32
  image_width: int = 32
  image_channels: int = 3
  image_mean: tuple[float, ...] = (127.5, 127.5, 127.5)
  image_std: tuple[float, ...] = (127.5, 127.5, 127.5)


class ImageProcessorTest(parameterized.TestCase):
  # [CLASS DOC]
  # Tests preprocessing of single images and batching/padding behavior of ImageProcessor.

  def setUp(self):
    # [FUNCTION DOC]
    # HELPER: fixture setting height, width (32) and channels (3) and building an ImageProcessor from a
    # DummyConfig with those values.
    super().setUp()
    self.height = 32
    self.width = 32
    self.channels = 3
    config = DummyConfig(
        image_height=self.height,
        image_width=self.width,
        image_channels=self.channels,
    )
    self.processor = image_processor.ImageProcessor(config)

  def _create_dummy_image_file(self, filename='test_image.png'):
    # [FUNCTION DOC]
    # HELPER: writes a 100x100 black RGB image to a new temporary directory under the given filename
    # (default "test_image.png"), registers removal of the directory as cleanup, and returns the file path.
    img_array = np.zeros((100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)

    # TODO(abheesht17): Use self.create_tempdir(). It was failing on GitHub CI,
    # but revisit.
    temp_dir = tempfile.mkdtemp()
    self.addCleanup(lambda: shutil.rmtree(temp_dir))

    temp_file = os.path.join(temp_dir, filename)
    img.save(temp_file)
    return temp_file

  def test_process_none_image(self):
    # [FUNCTION DOC]
    # Verifies preprocess_image(None) returns a zero array of shape (height, width, 3).
    processed_image = self.processor.preprocess_image(None)
    np.testing.assert_array_equal(
        processed_image, np.zeros((self.height, self.width, 3))
    )

  def test_path_input(self):
    # [FUNCTION DOC]
    # Verifies preprocess_image on a file path loads, resizes and normalizes a black image to all -1.0.
    img_path = self._create_dummy_image_file()
    processed_image = self.processor.preprocess_image(img_path)
    np.testing.assert_allclose(
        processed_image, -1.0 * np.ones((self.height, self.width, 3))
    )

  def test_array_input(self):
    # [FUNCTION DOC]
    # Verifies preprocess_image on a uint8 numpy array resizes and normalizes a black image to all -1.0.
    img_array = np.zeros((100, 100, 3), dtype=np.uint8)
    processed_image = self.processor.preprocess_image(img_array)
    np.testing.assert_allclose(
        processed_image, -1.0 * np.ones((self.height, self.width, 3))
    )

  @parameterized.product(
      input_type=['array', 'path'],
      is_dim_0=[True, False],
  )
  def test_single_image(self, input_type, is_dim_0):
    # [FUNCTION DOC]
    # Verifies calling the processor with one image (array or path), given either bare or wrapped in a
    # list, yields one batch element with one normalized image.
    if input_type == 'array':
      images = np.zeros((100, 100, 3), dtype=np.uint8)
    elif input_type == 'path':
      images = self._create_dummy_image_file()
    else:
      raise ValueError(f'Invalid input_type: {input_type}')

    if not is_dim_0:
      images = [images]

    processed_images = self.processor(images=images)
    self.assertLen(processed_images, 1)
    self.assertLen(processed_images[0], 1)
    np.testing.assert_allclose(
        processed_images[0][0], -1.0 * np.ones((self.height, self.width, 3))
    )

  def test_multiple_images_dim_1(self):
    # [FUNCTION DOC]
    # Verifies a flat list of two differently sized arrays is treated as two batch elements, each holding
    # one normalized image.
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img2 = np.zeros((50, 50, 3), dtype=np.uint8)
    images = [img1, img2]
    processed_images = self.processor(images=images)
    np.testing.assert_allclose(
        processed_images[0][0], -1.0 * np.ones((self.height, self.width, 3))
    )
    np.testing.assert_allclose(
        processed_images[1][0], -1.0 * np.ones((self.height, self.width, 3))
    )

  @parameterized.named_parameters(
      {'testcase_name': 'all_dim_1', 'input_type': 'all_dim_1'},
      {'testcase_name': 'mixed', 'input_type': 'mixed'},
  )
  def test_padding(self, input_type):
    # [FUNCTION DOC]
    # Verifies padding: for a batch whose elements have one and two images (given as nested lists or as a
    # bare image plus a list), the first element is padded with a zero image to length two while the
    # second element holds two normalized images.
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img2 = np.zeros((50, 50, 3), dtype=np.uint8)

    if input_type == 'all_dim_1':
      images = [[img1], [img1, img2]]
    else:
      images = [img1, [img1, img2]]

    processed_images = self.processor(images=images)
    np.testing.assert_allclose(
        processed_images[0][0], -1.0 * np.ones((self.height, self.width, 3))
    )
    # Padded image should be zeros
    np.testing.assert_allclose(
        processed_images[0][1], np.zeros((self.height, self.width, 3))
    )
    np.testing.assert_allclose(
        processed_images[1][0], -1.0 * np.ones((self.height, self.width, 3))
    )
    np.testing.assert_allclose(
        processed_images[1][1], -1.0 * np.ones((self.height, self.width, 3))
    )

  def test_mixed_inputs(self):
    # [FUNCTION DOC]
    # Verifies the same padding behavior when mixing a numpy array and a file path within a batch.
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img2 = self._create_dummy_image_file()
    images = [img1, [img1, img2]]
    processed_images = self.processor(images=images)
    np.testing.assert_allclose(
        processed_images[0][0], -1.0 * np.ones((self.height, self.width, 3))
    )
    np.testing.assert_allclose(
        processed_images[0][1], np.zeros((self.height, self.width, 3))
    )
    np.testing.assert_allclose(
        processed_images[1][0], -1.0 * np.ones((self.height, self.width, 3))
    )
    np.testing.assert_allclose(
        processed_images[1][1], -1.0 * np.ones((self.height, self.width, 3))
    )

  def test_call_with_none_in_batch(self):
    # [FUNCTION DOC]
    # Verifies a None batch element becomes a zero image while the other element is normalized normally.
    images = [None, [np.zeros((100, 100, 3), dtype=np.uint8)]]
    processed_images = self.processor(images=images)
    np.testing.assert_allclose(
        processed_images[0][0], np.zeros((self.height, self.width, 3))
    )
    np.testing.assert_allclose(
        processed_images[1][0], -1.0 * np.ones((self.height, self.width, 3))
    )


if __name__ == '__main__':
  absltest.main()
