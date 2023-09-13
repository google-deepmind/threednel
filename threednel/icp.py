"""Module containing ICP-related utils."""
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
from typing import Callable

import numba
import numpy as np
from scipy.spatial import KDTree


def apply_transform(coords: np.array, transform: np.ndarray) -> np.ndarray:
  """Apply transformation matrix to coordinates.

  Args:
      coords: Array of shape (..., 3)
      transform: Array of shape (4, 4)

  Returns:
      np.ndarray: Array of shape (..., 3), transformed coordinates.
  """
  coords = np.einsum(
      'ij,...j->...i',
      transform,
      np.concatenate([coords, np.ones(coords.shape[:-1] + (1,))], axis=-1),
  )[..., :-1]
  return coords


@numba.jit(nopython=True, cache=True, nogil=True)
def get_transform_from_cloud_to_target(
    A: np.ndarray, B: np.ndarray
) -> np.ndarray:
  """Estimate the best rigid transform to transform a point cloud A to a point cloud B.

  Args:
    A: A source point cloud.
    B: A target point cloud.

  Returns:
    The estimated rigid transform.
  """
  assert A.shape == B.shape

  # find mean column wise
  centroid_A = np.array(
      [[np.mean(A[:, 0]), np.mean(A[:, 1]), np.mean(A[:, 2])]]
  )
  centroid_B = np.array(
      [[np.mean(B[:, 0]), np.mean(B[:, 1]), np.mean(B[:, 2])]]
  )

  # subtract mean
  Am = A - centroid_A
  Bm = B - centroid_B

  H = np.transpose(Am) @ Bm

  # find rotation
  U, S, Vt = np.linalg.svd(H)
  R = Vt.T @ U.T

  # special reflection case
  if np.linalg.det(R) < 0:
    Vt[2, :] *= -1
    R = Vt.T @ U.T

  t = -R @ np.transpose(centroid_A) + np.transpose(centroid_B)

  transform = np.eye(4)
  transform[:3, :3] = R
  transform[:3, 3] = t.ravel()
  return transform


@dataclass
class ICP:
  """Class implementing iterative closest point (ICP).

  target_cloud: Array of shape (n_target_points, 3). Target point cloud.
  render: function.
      Args:
          pose: Array of shape (4, 4). Pose of the object.
          obj_idx: object index in the scene.

      Returns:
          cloud: Array of shape (n_points, 3). Rendered point cloud of the
          object.
          model_mask: Array of shape (H, W). Model mask.
  n_outer_iterations: Number of outer iterations.
      We call the renderer once in each outer iteration.
  n_inner_iterations: Number of inner iterations.
      We update the rendered cloud instead of call the renderer in each inner
      iteration.
  """

  target_cloud: np.ndarray
  render: Callable
  n_outer_iterations: int = 5
  n_inner_iterations: int = 2

  def __post_init__(self):
    self.target_cloud = np.array(self.target_cloud)
    self.target_tree = KDTree(self.target_cloud)

  def _update_pose_and_cloud(self, pose: np.ndarray, cloud: np.ndarray):
    """Locally update object pose and point cloud without re-rendering."""
    _, idxs = self.target_tree.query(cloud)
    target_neighbors = self.target_cloud[idxs, :]
    transform = get_transform_from_cloud_to_target(
        cloud.astype(np.float32), target_neighbors.astype(np.float32)
    )
    pose = transform.dot(pose)
    cloud = apply_transform(cloud, transform)
    return pose, cloud

  def _update_pose(self, pose: np.ndarray, obj_idx: int):
    """Render and run ICP to update object pose."""
    cloud = self.render(pose, obj_idx)[0]
    for _ in range(self.n_inner_iterations):
      pose, cloud = self._update_pose_and_cloud(pose, cloud)

    return pose

  def fit(self, pose: np.ndarray, obj_idx: int):
    """Function to run ICP."""
    for _ in range(self.n_outer_iterations):
      pose = self._update_pose(pose, obj_idx)

    return pose
