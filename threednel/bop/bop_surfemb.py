"""Module containing a wrapper class for a pretrained SurfEMB model."""
# Copyright 2023 DeepMind Technologies Limited
# Copyright 2023 Massachusetts Institute of Technology (M.I.T.)
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
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import jax.dlpack
import jax.numpy as jnp
from jax.scipy.special import expit
import numpy as np
from threednel.bop.data import RGBDImage
from threednel.third_party.surfemb.surface_embedding import SurfaceEmbeddingModel
import torch


@dataclass
class BOPSurfEmb:
  """Wrapper class around a pretrained SurfEMB model.

  Supports applying a pretrained SurfEMB model to a given RGB image to get query
  embeddings and mask estimations.

  Attributes:
    surfemb_model_path: Path to the a SurfEMB model checkpoint trained on YCB
      objects.
    device: Device to use for running the SurfEMB model. Defaults to 'cuda:0'.
    surfemb_model: The loaded SurfEMB model.
    bop_obj_indices: Indices of all the relevant objects in the BOP data format.
      For the YCB-V dataset this refers to the object indices in
      https://bop.felk.cvut.cz/media/data/bop_datasets/ycbv_models.zip
  """

  surfemb_model_path: str
  device: str = 'cuda:0'

  def __post_init__(self):
    print(f'Loading surfemb model {self.surfemb_model_path} for dataset ycb.')
    device = torch.device(self.device)
    surfemb_model = SurfaceEmbeddingModel.load_from_checkpoint(
        self.surfemb_model_path
    )
    surfemb_model.eval()
    surfemb_model.freeze()
    surfemb_model.to(device)
    self.surfemb_model = surfemb_model
    self.bop_obj_indices = np.arange(1, 22)
    self.surfemb_model.init_bop_obj_indices_to_obj_indices(self.bop_obj_indices)

  def get_query_embeddings(
      self,
      img: RGBDImage,
      scale: float = 1.0,
      target_shape: Optional[Tuple[int, int]] = None,
      bop_obj_indices: Optional[np.ndarray] = None,
  ):
    """Apply the SurfEMB model on an RGBDImage to get query embeddings.

    Args:
        img: An RGB-D image
        scale: scaling to apply to the RGB image. Defaults to 1.0. The width and
          height of the RGB image have to be multiples of 32 in order to be
          compatible with the CNNs.
        target_shape: (width, height) of the final query embeddings output.
        bop_obj_indices: Optional array specifying the indices of the set of
          objects for which we want to get query embeddings for. If None, return
          the query embeddings for all the objects specified in
          img.bop_obj_indices.

    Returns:
        query_embeddings: Array of shape target_shape + (n_bop_objects, emb_dim)
          The query embeddings for the relevant objects as specified in
          bop_obj_indices.
    """
    rgb = img.rgb.copy()
    if target_shape is None:
      target_shape = img.rgb.shape[:2]

    if bop_obj_indices is None:
      bop_obj_indices = img.bop_obj_indices
    else:
      for bop_obj_idx in bop_obj_indices:
        assert bop_obj_idx in img.bop_obj_indices

    dsize = (np.round(scale * np.array(rgb.shape[:2]) / 32).astype(int) * 32)[
        [1, 0]
    ]
    rgb = cv2.resize(rgb, dsize=dsize, interpolation=1)
    surfemb_model = self.surfemb_model
    query_embeddings = np.zeros(
        target_shape + (len(bop_obj_indices), surfemb_model.emb_dim)
    )
    for ii, bop_obj_idx in enumerate(bop_obj_indices):
      query_embeddings[:, :, ii] = cv2.resize(
          surfemb_model.infer_cnn(rgb, bop_obj_idx, False)[1].cpu().numpy(),
          dsize=(target_shape[1], target_shape[0]),
          interpolation=0,
      )

    return query_embeddings

  def get_query_embeddings_masks(
      self, img: RGBDImage, squeeze: bool = False
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Function to get both query embeddings and mask predictions.

    Args:
      img: The input RGBDImage.
      squeeze: Get rid of the dimension corresponding to the objects if there is
        only one object.

    Returns:
      The query embeddings and mask probabilities predictions.
    """
    surfemb_model = self.surfemb_model
    query_embeddings = []
    mask_lgts = []
    for bop_obj_idx in img.bop_obj_indices:
      mask_lgts_for_obj, query_embeddings_for_obj = surfemb_model.infer_cnn(
          img.rgb, bop_obj_idx, False
      )
      query_embeddings.append(
          jax.dlpack.from_dlpack(
              torch.to_dlpack(query_embeddings_for_obj.contiguous())
          )
      )
      mask_lgts.append(
          jax.dlpack.from_dlpack(
              torch.to_dlpack(mask_lgts_for_obj.contiguous())
          )
      )

    query_embeddings = jnp.stack(query_embeddings, axis=2)
    mask_lgts = jnp.stack(mask_lgts, axis=2)
    masks = expit(mask_lgts)
    if squeeze:
      query_embeddings = jnp.squeeze(query_embeddings)
      masks = jnp.squeeze(masks)

    return query_embeddings, masks
