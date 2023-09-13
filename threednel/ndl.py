"""Module implementing 3DNEL evaluation."""
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
import functools
import os
from dataclasses import dataclass
from typing import Tuple

import jax
import jax.dlpack
import jax.numpy as jnp
import numpy as np
import torch
from jax.scipy.special import logsumexp

from threednel.bop.data import RGBDImage
from threednel.third_party.surfemb.surface_embedding import \
    SurfaceEmbeddingModel


@dataclass
class JAXNDL:
  """JAX-based implementation of 3DNEL evaluation."""

  model: SurfaceEmbeddingModel
  n_objs: int
  r: float
  outlier_prob: float
  outlier_volume: float
  outlier_scaling: float
  filter_shape: Tuple[int, int]
  data_directory: str = os.environ['BOP_DATA_DIR']

  def __post_init__(self):
    self.p_background = self.outlier_prob / self.outlier_volume * self.outlier_scaling
    self.indices = list(range(self.n_objs))

    @functools.partial(
        jnp.vectorize,
        signature='(h,w,l),(h,w,m),(h,w)->()',
        excluded=(3, 4, 5, 6),
    )
    def ndl_from_rendered_data(
        model_xyz: jnp.ndarray,
        key_embeddings: jnp.ndarray,
        obj_ids: jnp.ndarray,
        data_xyz: jnp.ndarray,
        query_embeddings: jnp.ndarray,
        log_normalizers: jnp.ndarray,
        data_mask: jnp.ndarray,
    ):
      model_mask = obj_ids >= 0
      p_foreground = (1.0 - self.outlier_prob) / model_mask.sum()
      log_prob = neural_embedding_likelihood(
          data_xyz=data_xyz,
          query_embeddings=query_embeddings,
          log_normalizers=log_normalizers,
          model_xyz=model_xyz,
          key_embeddings=key_embeddings,
          model_mask=model_mask,
          obj_ids=obj_ids,
          data_mask=data_mask,
          r=self.r,
          p_background=self.p_background,
          p_foreground=p_foreground,
          filter_shape=self.filter_shape,
      )
      return log_prob

    self.ndl_from_rendered_data = jax.jit(ndl_from_rendered_data)

  def set_for_new_img(
      self,
      img: RGBDImage,
      query_embeddings: np.ndarray,
      log_normalizers: np.ndarray,
      data_mask: np.ndarray,
      scale_factor: float,
  ):
    """Image-specific updates to allow 3DNEL evaluations on a new image."""
    self.initialized = True
    assert len(img.bop_obj_indices) == self.n_objs
    self.renderer = img.get_renderer(self.data_directory, scale_factor)
    shape = (self.renderer.height, self.renderer.width)
    data_xyz = img.unproject()
    self.data_xyz = jax.image.resize(
        data_xyz,
        shape=shape + data_xyz.shape[2:],
        method='nearest',
    )
    self.query_embeddings = jax.image.resize(
        query_embeddings,
        shape=shape + query_embeddings.shape[2:],
        method='nearest',
    )
    self.log_normalizers = jax.image.resize(
        log_normalizers,
        shape=shape + log_normalizers.shape[2:],
        method='nearest',
    )
    self.data_mask = jax.image.resize(data_mask, shape=shape, method='nearest')
    self.bop_obj_indices = jax.device_put(np.array(img.bop_obj_indices))

  def compute_likelihood(
      self,
      poses: np.ndarray,
  ):
    """Wrapper function to compute likelihoods of a given set of poses.

    Args:
      poses: Array of shape (n_objs, n_particles, 4, 4) The set of pose
        hypotheses we are going to score

    Returns:
      The likelihoods of the given set of poses.
    """
    assert self.initialized
    rendered_data = self.renderer.render(poses, self.indices)
    key_embeddings = torch.zeros(
        rendered_data.obj_coords.shape[:-1] + (self.model.emb_dim,),
        device=self.model.device,
    )
    obj_coords_torch = torch.from_dlpack(jax.dlpack.to_dlpack(rendered_data.obj_coords))
    obj_ids_torch = torch.from_dlpack(jax.dlpack.to_dlpack(rendered_data.obj_ids))
    for ii in range(len(self.bop_obj_indices)):
      mask = obj_ids_torch == ii
      key_embeddings[mask] = self.model.infer_mlp(
          obj_coords_torch[mask], int(self.bop_obj_indices[ii])
      )

    key_embeddings = jax.dlpack.from_dlpack(torch.to_dlpack(key_embeddings))
    log_prob = self.ndl_from_rendered_data(
        rendered_data.model_xyz,
        key_embeddings,
        rendered_data.obj_ids,
        self.data_xyz,
        self.query_embeddings,
        self.log_normalizers,
        self.data_mask,
    )
    return log_prob


@functools.partial(jax.jit, static_argnames='filter_shape')
def neural_embedding_likelihood(
    data_xyz: jnp.ndarray,
    query_embeddings: jnp.ndarray,
    log_normalizers: jnp.ndarray,
    model_xyz: jnp.ndarray,
    key_embeddings: jnp.ndarray,
    model_mask: jnp.ndarray,
    obj_ids: jnp.ndarray,
    data_mask: jnp.ndarray,
    r: float,
    p_background: float,
    p_foreground: float,
    filter_shape: Tuple[int, int],
) -> jnp.ndarray:
  """Function implementing 3DNEL evalaution.

  Args:
      data_xyz: Array of shape (H, W, 3). Observed point cloud organized as an
        image.
      query_embeddings: Array of shape (H, W, n_objs, d). Query embeddings for
        each observed pixel using models from different objects.
      log_normalizers: Array of shape (H, W, n_objs). The log normalizers for
        each pixel given each object model
      model_xyz: Array of shape (H, W, 3). Rendered point cloud organized as an
        image.
      key_embeddings: Array of shape (H, W, d). Key embeddings organized as an
        image.
      model_mask: Array of shape (H, W). Mask indicating relevant pixels from
        rendering.
      obj_ids: Array of shape (H, W). The object id of each pixel.
      data_mask: Array of shape (H, W). Mask indicating the relevant set of
        pixels.
      r: Radius of the ball.
      p_background: background probability.
      p_foreground: foreground probability.
      filter_shape: used to restrict likelihood evaluation to a 2D neighborhood.

  Returns:
    The likelihood as evaluated using 3DNEL.
  """

  obj_ids = jnp.round(obj_ids).astype(jnp.int32)
  padding = [
      (filter_shape[ii] // 2, filter_shape[ii] - filter_shape[ii] // 2 - 1)
      for ii in range(len(filter_shape))
  ]
  model_xyz_padded = jnp.pad(model_xyz, pad_width=padding + [(0, 0)])
  key_embeddings_padded = jnp.pad(key_embeddings, pad_width=padding + [(0, 0)])
  model_mask_padded = jnp.pad(model_mask, pad_width=padding)
  obj_ids_padded = jnp.pad(obj_ids, pad_width=padding)

  @functools.partial(
      jnp.vectorize,
      signature='(m),(n),(o,d),(o)->()',
  )
  def log_likelihood_for_pixel(
      ij: jnp.ndarray,
      data_xyz_for_pixel: jnp.ndarray,
      query_embeddings_for_pixel: jnp.ndarray,
      log_normalizers_for_pixel: jnp.ndarray,
  ):
    """Function to evaluate the log-likelihood for a given pixel.

    Args:
      ij: Array of shape (2,). The i, j index of the pixel.
      data_xyz_for_pixel: The camera frame coordinate of the point at the pixel.
      query_embeddings_for_pixel: The query embeddings at the pixel.
      log_normalizers_for_pixel: The log_normalizers at the pixel.

    Returns:
      The log likelihood for the given pixel.
    """
    model_xyz_patch = jax.lax.dynamic_slice(
        model_xyz_padded,
        jnp.array([ij[0], ij[1], 0]),
        (filter_shape[0], filter_shape[1], 3),
    )
    key_embeddings_patch = jax.lax.dynamic_slice(
        key_embeddings_padded,
        jnp.array([ij[0], ij[1], 0]),
        (filter_shape[0], filter_shape[1], key_embeddings.shape[-1]),
    )
    model_mask_patch = jax.lax.dynamic_slice(model_mask_padded, ij, filter_shape)
    obj_ids_patch = jax.lax.dynamic_slice(obj_ids_padded, ij, filter_shape)
    log_prob_correspondence = (
        jnp.sum(
            query_embeddings_for_pixel[obj_ids_patch] * key_embeddings_patch,
            axis=-1,
        )
        - log_normalizers_for_pixel[obj_ids_patch]
    ).ravel()
    distance = jnp.linalg.norm(data_xyz_for_pixel - model_xyz_patch, axis=-1).ravel()
    a = jnp.concatenate([jnp.zeros(1), log_prob_correspondence])
    b = jnp.concatenate([
        jnp.array([p_background]),
        jnp.where(
            jnp.logical_and(distance <= r, model_mask_patch.ravel() > 0),
            3 * p_foreground / (4 * jnp.pi * r**3),
            0.0,
        ),
    ])
    log_mixture_prob = logsumexp(a=a, b=b)
    return log_mixture_prob

  log_mixture_prob = log_likelihood_for_pixel(
      jnp.moveaxis(jnp.mgrid[: data_xyz.shape[0], : data_xyz.shape[1]], 0, -1),
      data_xyz,
      query_embeddings,
      log_normalizers,
  )
  return jnp.sum(jnp.where(data_mask, log_mixture_prob, 0.0))
