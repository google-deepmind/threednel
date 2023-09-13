"""Module containing the class for pose hypotheses generation."""
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
from typing import Sequence, Tuple

import jax
import jax.dlpack
import jax.numpy as jnp
import numpy as np
import torch
from threednel.bop.bop_surfemb import BOPSurfEmb
from threednel.bop.bop_vote import BOPVoting
from threednel.bop.data import RGBDImage


@dataclass
class HypothesesGeneration:
  """Class for pose hypotheses generation.

  Attributes:
    bop_surfemb: Module for interacting with a pretrained SurfEMB model.
    bop_vote: Module implementing the spherical voting process.
    mask_threshold: Threshold for mask probabilities. Used to mask out
      irrevelant, noisy query embeddings.
    n_top_translations: Number of top translations to look at for pose
      hypotheses generation.
    n_pose_hypotheses_per_crop: Number of top pose hypotheses to keep for each
      2D detector crop.
    maximum_filter_size: Maximum filter size for doing non-max suppression.
    n_top_rotations_per_translation: Number of top rotations to look at per
      translation for the pose hypotheses generation process.
    n_pose_hypotheses_per_object: Number of top pose hypotheses to keep for each
      object.
    default_scale: The scale at which to obtain the query embeddings.
    use_crops: Whether to use information from 2D detection to improve pose
      hypotheses generation.
  """

  bop_surfemb: BOPSurfEmb
  bop_vote: BOPVoting
  mask_threshold: float = 0.7
  n_top_translations: int = 20
  n_pose_hypotheses_per_crop: int = 80
  maximum_filter_size: int = 10
  n_top_rotations_per_translation: int = 10
  n_pose_hypotheses_per_object: int = 30
  default_scale: float = 1.5
  use_crops: bool = True

  def _process_crop(self, crop_img: RGBDImage):
    """Process a detector crop from 2D detection.

    Args:
      crop_img: A detector crop from 2D detection structured as an RGBDImage.

    Returns:
      Query embeddings, top-scoring pose hypotheses and their scores.
    """
    (
        query_embeddings,
        masks,
    ) = self.bop_surfemb.get_query_embeddings_masks(crop_img)
    masks = jnp.logical_and(crop_img.depth > 0, masks[..., 0] > self.mask_threshold)
    query_embeddings = query_embeddings * masks[..., None, None]
    voting_voxel_grids = self.bop_vote.get_voting_voxel_grids(
        crop_img,
        torch.from_dlpack(jax.dlpack.to_dlpack(query_embeddings)),
        mask=torch.from_dlpack(jax.dlpack.to_dlpack(masks)),
    )
    pose_hypotheses, top_scores = self.bop_vote.get_top_pose_hypotheses(
        voting_voxel_grids,
        crop_img.bop_obj_indices,
        n_top_translations=self.n_top_translations,
        n_pose_hypotheses=self.n_pose_hypotheses_per_crop,
        return_scores=True,
    )
    return query_embeddings, pose_hypotheses, top_scores

  def _combine_query_embeddings_from_crops(
      self,
      img: RGBDImage,
      obj_idx: int,
      query_embeddings_list: Sequence[jnp.ndarray],
  ):
    """Function to combine query embeddings from multiple detector crops for the same object.

    We rescale the query embedding maps from each detector crop, and use the
    estimated masks to merge the query embedding maps from multiple detector
    crops into a single query embedding map.
    When the masks from multiple detector crops overlap, we take the query
    embedding with the maximum L2 norm.
    """
    query_embeddings_for_obj = jnp.zeros(
        img.rgb.shape[:2]
        + (
            len(query_embeddings_list),
            self.bop_surfemb.surfemb_model.emb_dim,
        )
    )
    for crop_idx in range(len(query_embeddings_list)):
      left, top, right, bottom = img.annotations[obj_idx]['detector_crops'][crop_idx][
          'AABB_crop'
      ]
      top_padding = max(top, 0) - top
      left_padding = max(left, 0) - left
      bottom_padding = bottom - min(query_embeddings_for_obj.shape[0], bottom)
      bottom_idx = -bottom_padding if bottom_padding > 0 else None
      right_padding = right - min(query_embeddings_for_obj.shape[1], right)
      right_idx = -right_padding if right_padding > 0 else None
      query_embeddings_for_obj = query_embeddings_for_obj.at[
          top + top_padding : bottom - bottom_padding,
          left + left_padding : right - right_padding,
          [crop_idx],
      ].set(
          jax.image.resize(
              query_embeddings_list[crop_idx],
              shape=(bottom - top, right - left)
              + (1, self.bop_surfemb.surfemb_model.emb_dim),
              method='nearest',
          )[top_padding:bottom_idx, left_padding:right_idx]
      )

    query_embeddings_for_obj = query_embeddings_for_obj[
        jnp.arange(img.rgb.shape[0])[:, None],
        jnp.arange(img.rgb.shape[1])[None],
        jnp.argmax(jnp.linalg.norm(query_embeddings_for_obj, axis=-1), axis=-1),
    ][:, :, None]
    return query_embeddings_for_obj

  def generate_from_crops_for_obj(
      self, img: RGBDImage, obj_idx: int
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate pose hypotheses from the available detector crops."""
    pose_hypotheses_list = []
    scores_list = []
    bop_obj_idx = img.bop_obj_indices[obj_idx]
    assert img.annotations[obj_idx].get('detector_crops', [])
    query_embeddings_list = []
    for detector_crop in img.annotations[obj_idx]['detector_crops']:
      crop_img = RGBDImage(
          rgb=detector_crop['rgb_crop'],
          depth=detector_crop['depth_crop'],
          intrinsics=detector_crop['K_crop'],
          bop_obj_indices=np.array([bop_obj_idx]),
          fill_in_depth=img.fill_in_depth,
          max_depth=img.max_depth,
      )
      (
          query_embeddings_for_crop,
          pose_hypotheses,
          top_scores,
      ) = self._process_crop(crop_img)
      query_embeddings_list.append(query_embeddings_for_crop)
      pose_hypotheses_list.append(pose_hypotheses[0])
      scores_list.append(top_scores[0])

    query_embeddings_for_obj = self._combine_query_embeddings_from_crops(
        img, obj_idx, query_embeddings_list
    )
    pose_hypotheses_for_obj = jnp.concatenate(pose_hypotheses_list, axis=0)[
        jnp.argsort(-jnp.concatenate(scores_list))
    ]
    return query_embeddings_for_obj, pose_hypotheses_for_obj

  def generate_from_whole_image_for_obj(self, img: RGBDImage, obj_idx: int):
    """Generate pose hypotheses from the the whole image."""
    bop_obj_idx = img.bop_obj_indices[obj_idx]
    query_embeddings_for_obj = jax.device_put(
        self.bop_surfemb.get_query_embeddings(
            img,
            scale=self.default_scale,
            target_shape=img.rgb.shape[:2],
            bop_obj_indices=np.array([bop_obj_idx]),
        )
    )
    query_embeddings_for_obj = (
        query_embeddings_for_obj * (img.depth > 0)[..., None, None]
    )
    voting_voxel_grids = self.bop_vote.get_voting_voxel_grids(
        img,
        torch.from_dlpack(jax.dlpack.to_dlpack(query_embeddings_for_obj)),
        mask=torch.from_dlpack(jax.dlpack.to_dlpack(jax.device_put(img.depth > 0))),
        bop_obj_indices=np.array([bop_obj_idx]),
    )
    pose_hypotheses_for_obj = self.bop_vote.get_top_pose_hypotheses_non_max_suppression(
        voting_voxel_grids[0],
        bop_obj_idx,
        maximum_filter_size=self.maximum_filter_size,
        n_top_rotations_per_translation=self.n_top_rotations_per_translation,
        n_pose_hypotheses=self.n_pose_hypotheses_per_object,
        return_scores=False,
    )
    return query_embeddings_for_obj, pose_hypotheses_for_obj

  def generate(self, img: RGBDImage):
    """Generate pose hypotheses for all objects in the scene.

    Uses detector crops from 2D detection when they are availbel, and falls back
    to using the whole image when 2D detection is not available.
    """
    query_embeddings_all = []
    pose_hypotheses_all = []
    for obj_idx in range(len(img.bop_obj_indices)):
      if self.use_crops and img.annotations[obj_idx].get('detector_crops', []):
        (
            query_embeddings_for_obj,
            pose_hypotheses_for_obj,
        ) = self.generate_from_crops_for_obj(img, obj_idx)
      else:
        (
            query_embeddings_for_obj,
            pose_hypotheses_for_obj,
        ) = self.generate_from_whole_image_for_obj(img, obj_idx)

      query_embeddings_all.append(query_embeddings_for_obj)
      pose_hypotheses_all.append(pose_hypotheses_for_obj)

    query_embeddings = jnp.concatenate(query_embeddings_all, axis=2)
    data_mask = jnp.logical_and(
        img.depth > 0,
        jnp.any(jnp.linalg.norm(query_embeddings, axis=-1) > 0, axis=-1),
    )
    return query_embeddings, data_mask, pose_hypotheses_all
