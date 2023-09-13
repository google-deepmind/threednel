# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from contextlib import contextmanager
import time
from typing import Sequence

import cv2
import numpy as np
from threednel.third_party.surfemb.surface_embedding import SurfaceEmbeddingModel
import torch


@contextmanager
def timer(text='', do=True):
  if do:
    start = time.time()
    try:
      yield
    finally:
      print(f'{text}: {time.time() - start:.4}s')
  else:
    yield


def get_key_embeddings_visualizations(
    key_embeddings: np.ndarray,
    obj_ids: np.ndarray,
    all_key_embeddings: Sequence[np.ndarray],
    model: SurfaceEmbeddingModel,
) -> np.ndarray:
  """get_key_embeddings_visualizations.

  Args:
      key_embeddings (np.ndarray): Array of shape (H, W, emb_dim) Descriptors
        for different model
      obj_ids (np.ndarray): Array of shape (H, W) Object ids within the scene.
        Ranges from -1 (background) to n_objs - 1
      all_key_embeddings (Sequence[np.ndarray]): all_key_embeddings Sequence of
        length n_objs. Descriptors associated with the different points sampled
        from the object surface.
      model (SurfaceEmbeddingModel): model

  Returns:
      Array of shape (H, W, 3). Visualization of the model descriptors.
  """
  key_embeddings_visualizations = np.zeros(key_embeddings.shape[:-1] + (3,))
  for ii, key_embeddings in enumerate(all_key_embeddings):
    key_embeddings_visualizations[obj_ids == ii] = (
        model.get_emb_vis(
            torch.from_numpy(key_embeddings[obj_ids == ii]),
            demean=torch.from_numpy(key_embeddings).mean(dim=0),
        )
        .cpu()
        .numpy()
    )

  return key_embeddings_visualizations


def get_bbox_rgb_crop_K_crop(
    rgb: np.ndarray,
    depth: np.ndarray,
    bbox: np.ndarray,
    K: np.ndarray,
    crop_scale: float = 1.2,
    crop_res: int = 224,
):
  """Adapted from https://github.com/rasmushaugaard/surfemb/blob/53e1852433a3b2b84fedc7a3a01674fe1b6189cc/surfemb/data/std_auxs.py#L60

  Args:
      rgb (np.ndarray): Full RGB image
      bbox (np.ndarray): Array of shape (4,). Bounding box of the detector crop.
        4 elements are left, top, right, bottom
      crop_scale (float): crop_scale
      crop_res (int): crop_res

  Returns:
      crop_bbox_in_original: Tuple of length 4. Bounding box in the original
      full RGB image.
          4 elements are left, top, right, bottom
      rgb_crop: Array of shape (crop_res, crop_res). Cropped RGB image
  """
  R = np.eye(2)
  left, top, right, bottom = bbox
  cy, cx = (top + bottom) / 2, (left + right) / 2
  size = crop_res / max(bottom - top, right - left) / crop_scale
  r = crop_res
  M = np.concatenate((R, [[-cx], [-cy]]), axis=1) * size
  M[:, 2] += r / 2
  Ms = np.concatenate((M, [[0, 0, 1]]))
  # calculate axis aligned bounding box in the original image of the rotated crop
  crop_corners = np.array(((0, 0, 1), (0, r, 1), (r, 0, 1), (r, r, 1))) - (
      0.5,
      0.5,
      0,
  )  # (4, 3)
  crop_corners = np.linalg.inv(Ms) @ crop_corners.T  # (3, 4)
  crop_corners = crop_corners[:2] / crop_corners[2:]  # (2, 4)
  left, top = np.floor(crop_corners.min(axis=1)).astype(int)
  right, bottom = np.ceil(crop_corners.max(axis=1)).astype(int) + 1
  crop_bbox_in_original = left, top, right, bottom
  rgb_crop = cv2.warpAffine(rgb, M, (r, r), flags=1)
  depth_crop = cv2.warpAffine(depth, M, (r, r), flags=0)
  K_crop = Ms @ K
  return crop_bbox_in_original, rgb_crop, depth_crop, K_crop
