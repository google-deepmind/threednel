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
import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import jax.dlpack
import jax.numpy as jnp
import numpy as np
import taichi as ti
import torch
import trimesh
from scipy.ndimage import maximum_filter
from threednel.bop.data import RGBDImage
from threednel.rotation import generate_prototype_rotations
from threednel.third_party.surfemb.surface_embedding import \
    SurfaceEmbeddingModel
from torch_cluster import fps
from tqdm import tqdm


@ti.kernel
def taichi_compute_max_indices_log_normalizers_probs(
    query_embeddings: ti.types.ndarray(element_dim=1),
    key_embeddings: ti.types.ndarray(element_dim=1),
    max_indices: ti.types.ndarray(dtype=ti.i32),
    log_normalizers: ti.types.ndarray(),
    probs: ti.types.ndarray(),
):
  """A Taichi kernel that updates max_indices, log_normalizers and probs in place.

  Args:
      query_embeddings: Array of shape (M, d). Query embeddings of the image
      key_embeddings: Array of shape (N, d). Key embeddings of the model points
      max_indices: Array of shape (M,). max_indices[i] = argmax(key_embeddings @
        query_embeddings[i]). Indices of the model points with key embeddings
        most similar to the given query embeddings.
      log_normalizers: Array of shape (M,). logsumexp(key_embeddings @
        query_embeddings[i]). Log of the normalization constant summing over all
        model points.
      probs: Array of shape (M,). probs[i] = np.exp(query_embeddings[i] *
        key_embeddings[max_indices[i]] - log_normalizers[i]). The probability
        (P_RGB) of the model point that is most similar to a given query
        embedding.
  """
  for i in range(query_embeddings.shape[0]):
    max_indices[i] = -1
    log_normalizers[i] = 0.0
    max_inner_product = -100000.0
    for k in range(key_embeddings.shape[0]):
      inner_product = key_embeddings[k].dot(query_embeddings[i])
      if inner_product <= max_inner_product:
        log_normalizers[i] += ti.exp(inner_product - max_inner_product)
      else:
        max_indices[i] = k
        log_normalizers[i] *= ti.exp(max_inner_product - inner_product)
        log_normalizers[i] += 1.0
        max_inner_product = inner_product

    log_normalizers[i] = ti.log(log_normalizers[i]) + max_inner_product

  for i in range(query_embeddings.shape[0]):
    probs[i] = ti.exp(
        key_embeddings[max_indices[i]].dot(query_embeddings[i]) - log_normalizers[i]
    )


@ti.kernel
def taichi_compute_log_normalizers(
    query_embeddings: ti.types.ndarray(element_dim=1),
    key_embeddings: ti.types.ndarray(element_dim=1),
    log_normalizers: ti.types.ndarray(),
):
  """A taichi kernel that updates log_normalizers in place.

  Args:
      query_embeddings: Array of shape (M, d). Query embeddings of the image
      key_embeddings: Array of shape (N, d). Key embeddings of the model points
      log_normalizers: Array of shape (M,). logsumexp(key_embeddings @
        query_embeddings[i]). Log of the normalization constant summing over all
        model points.
  """
  for i in range(query_embeddings.shape[0]):
    log_normalizers[i] = 0.0
    max_inner_product = -100000.0
    for k in range(key_embeddings.shape[0]):
      inner_product = key_embeddings[k].dot(query_embeddings[i])
      if inner_product <= max_inner_product:
        log_normalizers[i] += ti.exp(inner_product - max_inner_product)
      else:
        log_normalizers[i] *= ti.exp(max_inner_product - inner_product)
        log_normalizers[i] += 1.0
        max_inner_product = inner_product

    log_normalizers[i] = ti.log(log_normalizers[i]) + max_inner_product


@ti.kernel
def taichi_spherical_vote(
    centers: ti.types.ndarray(element_dim=1),
    radiuses: ti.types.ndarray(),
    weights: ti.types.ndarray(),
    voxel_grid: ti.types.ndarray(),
    voxel_grid_start: ti.types.ndarray(element_dim=1),
    voxel_diameter: float,
    multipliers: ti.types.ndarray(),
):
  """A Taichi kernel that implements the spherical voting procedure.

  Updates voxel_grid in place. Units are milimeters.

  Refer to Section C in the supplementary for more details.

  Args:
      centers: Array of shape (batch_size, n_centers, 3,). Coordinates of the
        centers of the spheres.
      radiuses: Array of shape (batch_size, n_centers). Radiuses of the spheres.
      weights: Array of shape (batch_size, n_centers,). Weights of votes from
        the spheres.
      voxel_grid: Array of shape voxel_grid_shape. Votes from different sphere
        centers are aggregated into the voxel grid.
      voxel_grid_start: Array of shape (3,). Coordinate of the center of voxel
        (0, 0, 0).
      voxel_diameter: float. Diameter of a voxel.
      multipliers: Constant array with elements [1.0, -1.0].
  """
  for voxel in ti.grouped(voxel_grid):
    voxel_grid[voxel] = 0.0

  for ii, jj in centers:
    center_on_voxel_grid = (centers[ii, jj] - voxel_grid_start[None]) / voxel_diameter
    center_on_voxel_grid = ti.round(center_on_voxel_grid)
    radius_in_voxels = radiuses[ii, jj] / voxel_diameter + 0.5
    for x in range(ti.ceil(radius_in_voxels)):
      for y in range(ti.ceil(ti.sqrt(radius_in_voxels**2 - x**2))):
        z_range = (
            ti.ceil(
                ti.sqrt(
                    ti.max(
                        0.0,
                        (radiuses[ii, jj] / voxel_diameter - 0.5) ** 2
                        - x**2
                        - y**2,
                    )
                )
            ),
            ti.ceil(ti.sqrt(radius_in_voxels**2 - x**2 - y**2)),
        )
        for z in range(z_range[0], z_range[1]):
          for xx in range(2):
            if x == 0 and multipliers[xx] < 0:
              continue

            x_coord = ti.cast(
                center_on_voxel_grid[0] + multipliers[xx] * x,
                ti.i32,
            )
            if x_coord < 0 or x_coord >= voxel_grid.shape[1]:
              continue

            for yy in range(2):
              if y == 0 and multipliers[yy] < 0:
                continue

              y_coord = ti.cast(
                  center_on_voxel_grid[1] + multipliers[yy] * y,
                  ti.i32,
              )
              if y_coord < 0 or y_coord >= voxel_grid.shape[2]:
                continue

              for zz in range(2):
                if z == 0 and multipliers[zz] < 0:
                  continue

                z_coord = ti.cast(
                    center_on_voxel_grid[2] + multipliers[zz] * z,
                    ti.i32,
                )
                if z_coord < 0 or z_coord >= voxel_grid.shape[3]:
                  continue

                ti.atomic_add(
                    voxel_grid[ii, x_coord, y_coord, z_coord],
                    weights[ii, jj],
                )


@functools.partial(jax.jit, static_argnames=('n_top_translations', 'n_pose_hypotheses'))
def _get_top_pose_hypotheses(
    voting_voxel_grid: jnp.ndarray,
    keypoints_voxel_offsets: jnp.ndarray,
    voxel_grid_start: jnp.ndarray,
    voxel_diameter: float,
    prototype_rotations: jnp.ndarray,
    n_top_translations: int = 100,
    n_pose_hypotheses: int = 50,
):
  """Function to get top pose hypotheses given voting results in a voxel grid."""

  @functools.partial(jax.vmap, in_axes=0, out_axes=0)
  def get_top_pose_hypotheses_for_obj(
      voting_voxel_grid: jnp.ndarray,
      keypoints_voxel_offsets: jnp.ndarray,
  ):
    indices = jnp.array(
        jnp.unravel_index(
            jnp.argsort(-voting_voxel_grid[0].ravel()),
            voting_voxel_grid[0].shape,
        )
    ).T
    top_indices = indices[:n_top_translations]
    voxel_indices = top_indices[:, None, None] + keypoints_voxel_offsets
    valid_entries = jnp.logical_and(
        jnp.all(voxel_indices >= 0, axis=(-2, -1)),
        jnp.all(
            voxel_indices < jnp.array(voting_voxel_grid.shape[1:]),
            axis=(-2, -1),
        ),
    )
    scores = jnp.where(
        valid_entries,
        jnp.sum(
            voting_voxel_grid[1:][
                jnp.arange(keypoints_voxel_offsets.shape[1])[None, None],
                voxel_indices[..., 0],
                voxel_indices[..., 1],
                voxel_indices[..., 2],
            ],
            axis=-1,
        ),
        -jnp.inf,
    )
    top_voxel_indices, top_rotation_indices = jnp.unravel_index(
        jnp.argsort(-scores.ravel())[:n_pose_hypotheses], scores.shape
    )
    translations = voxel_grid_start + top_indices[top_voxel_indices] * voxel_diameter
    rotations = prototype_rotations[top_rotation_indices]
    transform_matrices = jnp.zeros((n_pose_hypotheses, 4, 4))
    transform_matrices = transform_matrices.at[:, :3, :3].set(rotations)
    transform_matrices = transform_matrices.at[:, :3, 3].set(translations)
    transform_matrices = transform_matrices.at[:, 3, 3].set(1.0)
    top_scores = scores[top_voxel_indices, top_rotation_indices]
    return transform_matrices.reshape((-1, 4, 4)), top_scores.ravel()

  return get_top_pose_hypotheses_for_obj(voting_voxel_grid, keypoints_voxel_offsets)


@dataclass
class BOPVoting:
  """Wrapper class for running the spherical voting process.

  The voting process assumes all the relevant object coordinates (x, y, z)
  satisfy np.all(np.array([x, y, z]) >= voxel_grid_start) and
  np.all(
    np.array([x, y, z]) <
    voxel_grid_start + voxel_diameter * np.array(voxel_grid_shape)
  )
  """

  data_directory: str
  num_points_on_sphere: int = 200
  num_inplane_rotations: int = 32
  principal_axis: np.ndarray = np.array([0, 0, 1.0])
  n_keypoints: int = 8
  voxel_diameter: float = 5.0
  voxel_grid_start: np.ndarray = np.array([-350.0, -210.0, 530.0], dtype=np.float32)
  voxel_grid_shape: Tuple[int, int, int] = (129, 87, 168)
  device: str = 'cuda:0'

  def __post_init__(self):
    device = torch.device(self.device)
    self.prototype_rotations = torch.from_numpy(
        generate_prototype_rotations(
            num_points_on_sphere=self.num_points_on_sphere,
            num_inplane_rotations=self.num_inplane_rotations,
            principal_axis=self.principal_axis,
        ).as_matrix()
    ).to(device)
    with open(
        os.path.join(
            self.data_directory,
            'bop',
            'ycbv',
            'models',
            'models_info.json',
        ),
        'r',
    ) as f:
      models_info = json.load(f)

    self.models_info = {
        int(bop_obj_idx): models_info[bop_obj_idx] for bop_obj_idx in models_info
    }
    self.bop_obj_indices = list(self.models_info.keys())
    self._init_voting_info()

  def _init_voting_info(self):
    """Initialize information relevant for the voting process.

    The relevant information includes:
      keypoints: Sampled on objects surfaces using farthest point sampling.
      model_radiuses: The distance of each model point to the sampled keypoints.
      keypoints_voxel_offsets: Offsets in terms of number of voxel grids, given
        model_radiuses and voxel_diameter.

    We precompute model_radiuses and keypoints_voxel_offsets to allow efficient
    implementation of the voting process.
    """
    device = torch.device(self.device)
    self.model_radiuses = {}
    self.keypoints = {}
    self.keypoints_voxel_offsets = {}
    for bop_obj_idx in tqdm(self.bop_obj_indices):
      model_coords = torch.from_numpy(
          np.asarray(
              trimesh.load(
                  os.path.join(
                      self.data_directory,
                      f'surface_samples/ycbv/obj_{bop_obj_idx:06d}.ply',
                  )
              ).vertices
          )
      ).to(device)
      keypoints_indices = fps(
          model_coords,
          batch=None,
          ratio=self.n_keypoints / model_coords.shape[0],
      )
      keypoints = model_coords[keypoints_indices]
      keypoints = torch.cat([torch.zeros((1, 3), device=device), keypoints], dim=0)
      self.keypoints[bop_obj_idx] = keypoints
      self.model_radiuses[bop_obj_idx] = torch.norm(
          model_coords - keypoints[:, None], dim=-1
      ).to(device)
      rotated_keypoints = torch.einsum(
          'ijk,mk->imj', self.prototype_rotations, keypoints[1:]
      )
      self.keypoints_voxel_offsets[bop_obj_idx] = torch.round(
          rotated_keypoints / self.voxel_diameter
      ).type(torch.int32)

  def load_all_key_embeddings(self, surfemb_model: SurfaceEmbeddingModel):
    """Loading the key embeddings for all models points in all objects."""
    print('Loading all key embeddings...')
    all_key_embeddings = {}
    for bop_obj_idx in tqdm(self.bop_obj_indices):
      verts_np = trimesh.load(
          os.path.join(
              self.data_directory,
              f'surface_samples/ycbv/obj_{bop_obj_idx:06d}.ply',
          )
      ).vertices
      mesh = trimesh.load(
          os.path.join(
              self.data_directory,
              f'bop/ycbv/models/obj_{bop_obj_idx:06d}.ply',
          )
      )
      offset, scale = (
          mesh.bounding_sphere.primitive.center,
          mesh.bounding_sphere.primitive.radius,
      )
      verts_norm = (verts_np - offset) / scale
      all_key_embeddings[bop_obj_idx] = surfemb_model.infer_mlp(
          torch.from_numpy(verts_norm).float().to(surfemb_model.device),
          bop_obj_idx,
      )

    self.all_key_embeddings = all_key_embeddings

  def get_max_indices_normalizers_probs(
      self,
      query_embeddings: torch.Tensor,
      bop_obj_indices: Union[np.ndarray, int],
      squeeze: bool = True,
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Function to get max_indices, log_normalizers and probs.

    Refer to taichi_compute_max_indices_log_normalizers_probs for details.

    Args:
        query_embeddings: Array of shape (N, len(bop_obj_indices),
          emb_dim)
        bop_obj_indices: indices should be in self.bop_obj_indices.
          Typically ranges from 1 to len(self.bop_obj_indices) + 1
        squeeze: Whether to get rid of redundant size 1 dimensions.

    Returns:
      max_indices, log_normalizers and probs.
    """
    target_shape = query_embeddings.shape[:-2]
    if query_embeddings.ndim == 4:
      query_embeddings = query_embeddings.reshape((-1,) + query_embeddings.shape[-2:])

    if np.isscalar(bop_obj_indices):
      bop_obj_indices = jnp.array([bop_obj_indices])

    assert query_embeddings.shape[1] == len(bop_obj_indices)
    max_indices = []
    log_normalizers = []
    probs = []
    for idx, bop_obj_idx in enumerate(bop_obj_indices):
      device = torch.device(self.device)
      max_indices_for_obj = torch.empty(
          query_embeddings.shape[:1], dtype=torch.int32, device=device
      )
      log_normalizers_for_obj = torch.empty(
          query_embeddings.shape[:1], dtype=torch.float32, device=device
      )
      probs_for_obj = torch.empty(
          query_embeddings.shape[:1], dtype=torch.float32, device=device
      )
      taichi_compute_max_indices_log_normalizers_probs(
          query_embeddings[:, idx].float(),
          self.all_key_embeddings[bop_obj_idx].float(),
          max_indices_for_obj,
          log_normalizers_for_obj,
          probs_for_obj,
      )
      ti.sync()
      max_indices.append(max_indices_for_obj)
      log_normalizers.append(log_normalizers_for_obj)
      probs.append(probs_for_obj)

    max_indices = torch.stack(max_indices, dim=1)
    max_indices = max_indices.reshape(target_shape + max_indices.shape[-1:])
    log_normalizers = torch.stack(log_normalizers, dim=1)
    log_normalizers = log_normalizers.reshape(target_shape + log_normalizers.shape[-1:])
    probs = torch.stack(probs, dim=1)
    probs = probs.reshape(target_shape + probs.shape[-1:])
    if squeeze:
      max_indices, log_normalizers, probs = (
          torch.squeeze(max_indices),
          torch.squeeze(log_normalizers),
          torch.squeeze(probs),
      )

    return max_indices, log_normalizers, probs

  def get_log_normalizers(
      self,
      query_embeddings: torch.Tensor,
      bop_obj_indices: Union[np.ndarray, int],
      squeeze: bool = True,
  ) -> torch.Tensor:
    """Function to get log_normalizers.

    Refer to taichi_compute_max_indices_log_normalizers_probs for details.

    Args:
        query_embeddings: Array of shape (N, len(bop_obj_indices),
          emb_dim)
        bop_obj_indices: indices should be in self.bop_obj_indices.
          Typically ranges from 1 to len(self.bop_obj_indices) + 1
        squeeze: Whether to get rid of redundant size 1 dimensions.

    Returns:
      log_normalizers.
    """
    target_shape = query_embeddings.shape[:-2]
    if query_embeddings.ndim == 4:
      query_embeddings = query_embeddings.reshape((-1,) + query_embeddings.shape[-2:])

    if np.isscalar(bop_obj_indices):
      bop_obj_indices = jnp.array([bop_obj_indices])

    assert query_embeddings.shape[1] == len(bop_obj_indices)
    log_normalizers = []
    for idx, bop_obj_idx in enumerate(bop_obj_indices):
      device = torch.device(self.device)
      log_normalizers_for_obj = torch.empty(
          query_embeddings.shape[:1], dtype=torch.float32, device=device
      )
      taichi_compute_log_normalizers(
          query_embeddings[:, idx].float().contiguous(),
          self.all_key_embeddings[bop_obj_idx].float(),
          log_normalizers_for_obj,
      )
      ti.sync()
      log_normalizers.append(log_normalizers_for_obj)

    log_normalizers = torch.stack(log_normalizers, dim=1)
    log_normalizers = log_normalizers.reshape(target_shape + log_normalizers.shape[-1:])
    if squeeze:
      log_normalizers = torch.squeeze(log_normalizers)

    return log_normalizers

  def get_voting_voxel_grids(
      self,
      img: RGBDImage,
      query_embeddings: torch.Tensor,
      mask: Optional[torch.Tensor] = None,
      bop_obj_indices: Optional[Union[jnp.ndarray, int]] = None,
      max_indices: Optional[torch.Tensor] = None,
      probs: Optional[torch.Tensor] = None,
      return_log_normalizers: bool = False,
  ):
    """Function to run voting and get the voting voxel grids.

    Args:
        img: The input RGBDImage
        query_embeddings: Array of shape (H, W, len(img.bop_obj_indices),
          emb_dim) or arrayay of shape (H, W, len(bop_obj_indices), emb_dim)
          when bop_obj_indices is not None
        mask: Array of shape (H, W)
        bop_obj_indices: bop_obj_indices
        max_indices: Optional precomputed max_indices
        probs: Optional precomputed probs
        return_log_normalizers: Whether to also return log_normalizers

    Returns:
        voting_voxel_grids: Array of shape (
            len(bop_obj_indices),
            self.n_keypoints + 1,
        ) + self.voxel_grid_shape
        and optionally the log_normalizers
    """
    device = torch.device(self.device)
    if np.isscalar(bop_obj_indices):
      bop_obj_indices = np.array([bop_obj_indices])

    if mask is None:
      mask = (torch.from_numpy(img.depth) > 0).to(device)

    assert mask.shape == query_embeddings.shape[:2]
    assert mask.shape == img.depth.shape
    data_xyz = torch.from_numpy(img.unproject()).to(device)[mask.type(torch.bool)]
    centers = torch.broadcast_to(data_xyz, (self.n_keypoints + 1,) + data_xyz.shape)

    if bop_obj_indices is None:
      bop_obj_indices = img.bop_obj_indices

    assert jnp.all(jnp.sum(bop_obj_indices[:, None] == img.bop_obj_indices, axis=1) > 0)
    obj_indices_in_img = torch.from_numpy(
        np.argmax(bop_obj_indices[:, None] == img.bop_obj_indices, axis=1)
    ).to(device)
    if max_indices is None or probs is None or return_log_normalizers:
      if query_embeddings.shape[2] == len(bop_obj_indices):
        (
            max_indices,
            log_normalizers_for_mask,
            probs,
        ) = self.get_max_indices_normalizers_probs(
            query_embeddings[mask.type(torch.bool)],
            bop_obj_indices,
            squeeze=False,
        )
      else:
        assert query_embeddings.shape[2] == len(img.bop_obj_indices)
        (
            max_indices,
            log_normalizers_for_mask,
            probs,
        ) = self.get_max_indices_normalizers_probs(
            query_embeddings[mask][:, obj_indices_in_img],
            bop_obj_indices,
            squeeze=False,
        )

      log_normalizers = torch.zeros(query_embeddings.shape[:-1], device=device)
      log_normalizers[mask.type(torch.bool)] = log_normalizers_for_mask

    voting_voxel_grids = torch.empty(
        (len(bop_obj_indices), self.n_keypoints + 1) + self.voxel_grid_shape,
        dtype=torch.float32,
        device=device,
    )
    multipliers = torch.tensor([1.0, -1.0], dtype=torch.float32, device=device)
    for idx, bop_obj_idx in enumerate(bop_obj_indices):
      weights = torch.broadcast_to(
          probs[:, idx],
          (self.n_keypoints + 1,) + probs[:, idx].shape,
      )
      radiuses = self.model_radiuses[bop_obj_idx][:, max_indices[:, idx]]
      taichi_spherical_vote(
          centers.float().contiguous(),
          radiuses.float().contiguous(),
          weights.float().contiguous(),
          voting_voxel_grids[idx],
          self.voxel_grid_start,
          self.voxel_diameter,
          multipliers,
      )
      ti.sync()

    if return_log_normalizers:
      return voting_voxel_grids, log_normalizers

    return voting_voxel_grids

  def get_top_pose_hypotheses(
      self,
      voting_voxel_grids: torch.Tensor,
      bop_obj_indices: np.ndarray,
      n_top_translations: int = 100,
      n_pose_hypotheses: int = 50,
      return_scores: bool = False,
  ):
    """Function to generate top-scoring pose hypotheses given voting results.

    Args:
      voting_voxel_grids: Voxel grids containing voting results.
      bop_obj_indices: Indices of the objects to generate pose hypotheses for.
      n_top_translations: Number of top translations we would look at.
      n_pose_hypotheses: Number of top poses hypotheses to generate.
      return_scores: Whether to additionally return the heuristic scores of the
        top pose hypotheses.

    Returns:
      Top pose hypotheses (in the form of 4x4 transform matrices) and optionally
      their heuristic scores.
    """
    if np.isscalar(bop_obj_indices):
      bop_obj_indices = np.array(bop_obj_indices)

    keypoints_voxel_offsets = torch.stack(
        [self.keypoints_voxel_offsets[bop_obj_idx] for bop_obj_idx in bop_obj_indices],
        dim=0,
    )
    transform_matrices, top_scores = _get_top_pose_hypotheses(
        jax.dlpack.from_dlpack(torch.to_dlpack(voting_voxel_grids)),
        jax.dlpack.from_dlpack(torch.to_dlpack(keypoints_voxel_offsets)),
        voxel_grid_start=self.voxel_grid_start,
        voxel_diameter=self.voxel_diameter,
        prototype_rotations=jax.dlpack.from_dlpack(
            torch.to_dlpack((self.prototype_rotations))
        ),
        n_top_translations=n_top_translations,
        n_pose_hypotheses=n_pose_hypotheses,
    )
    if return_scores:
      return transform_matrices, top_scores

    return transform_matrices

  def get_top_pose_hypotheses_non_max_suppression(
      self,
      voting_voxel_grid: torch.Tensor,
      bop_obj_idx: int,
      maximum_filter_size: int = 5,
      n_top_rotations_per_translation: int = 5,
      n_pose_hypotheses: int = 50,
      return_scores: bool = False,
  ):
    """Get top pose hypotheses with additional non-max suppression for translations."""
    voting_voxel_grid = voting_voxel_grid.cpu().numpy()
    max_filtered_voxel_grid = maximum_filter(
        voting_voxel_grid[0], size=maximum_filter_size
    )
    voting_grid_mask = np.logical_and(
        voting_voxel_grid[0] == max_filtered_voxel_grid,
        max_filtered_voxel_grid != 0,
    )
    top_indices = np.argwhere(voting_grid_mask)
    voxel_indices = (
        top_indices[:, None, None]
        + self.keypoints_voxel_offsets[bop_obj_idx].cpu().numpy()
    )
    valid_entries = np.logical_and(
        np.all(voxel_indices >= 0, axis=(-2, -1)),
        np.all(voxel_indices < np.array(voting_voxel_grid.shape[1:]), axis=(-2, -1)),
    )
    scores = np.zeros(valid_entries.shape)
    scores[valid_entries] = np.sum(
        voting_voxel_grid[1:][
            np.arange(self.n_keypoints)[None],
            voxel_indices[valid_entries][..., 0],
            voxel_indices[valid_entries][..., 1],
            voxel_indices[valid_entries][..., 2],
        ],
        axis=-1,
    )
    top_rotation_indices = np.argsort(-scores, axis=1)[
        :, :n_top_rotations_per_translation
    ]
    top_scores = scores[np.arange(scores.shape[0])[:, None], top_rotation_indices]
    scores_top_indices = np.array(
        np.unravel_index(np.argsort(-top_scores.ravel()), top_scores.shape)
    ).T[:n_pose_hypotheses]
    top_voxel_indices = scores_top_indices[:, 0]
    top_rotation_indices = top_rotation_indices[
        scores_top_indices[:, 0], scores_top_indices[:, 1]
    ]
    translations = (
        self.voxel_grid_start + top_indices[top_voxel_indices] * self.voxel_diameter
    )
    rotations = self.prototype_rotations[top_rotation_indices].cpu().numpy()
    transform_matrices = np.zeros((min(n_pose_hypotheses, translations.shape[0]), 4, 4))
    transform_matrices[:, :3, :3] = rotations
    transform_matrices[:, :3, -1] = translations
    transform_matrices[:, 3, 3] = 1.0
    if return_scores:
      top_scores = scores[top_voxel_indices, top_rotation_indices]
      return transform_matrices, top_scores

    return transform_matrices
