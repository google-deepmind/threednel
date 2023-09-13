"""Module implementing the inference program of 3DNEL MSIGP."""
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
import functools
from typing import Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation
from threednel.distributions import gaussian_vmf_sample
from threednel.icp import ICP


@functools.partial(jax.jit, static_argnames='n_samples')
def generate_candidate_poses(
    key: jnp.ndarray,
    curr_poses: jnp.ndarray,
    idx: int,
    var: jnp.ndarray,
    concentration: float,
    n_samples: int,
):
  """Randomly sample a set of candidate poses around the current poses.

  Args:
    key: PRNGKey
    curr_poses: Array of shape (n_objs, 4, 4). Current poses
    idx: Index of the object for which we are generating candidate poses
    var: Array of shape (3,). Diagonal for the covariance matrix.
    concentration: concentration. Parameter for Gaussian VMF
    n_samples: n_samples

  Returns:
    The splitted PRNGKey and the generated candidate poses.
  """
  keys = jax.random.split(key, n_samples + 1)
  key, subkeys = keys[0], keys[1:]
  sampled_poses = jax.vmap(
      gaussian_vmf_sample, in_axes=(0, None, None, None), out_axes=0
  )(subkeys, curr_poses[idx], var, concentration)
  candidate_poses = (
      jnp.tile(curr_poses[:, None], (1, n_samples, 1, 1))
      .at[idx]
      .set(sampled_poses)
  )
  return key, candidate_poses


def move_to_candidate_poses(
    inferred_poses: jnp.ndarray,
    candidate_poses: jnp.ndarray,
    compute_likelihood: Callable,
) -> jnp.ndarray:
  """Function to update the current poses given a set of candidate poses.

  We identify the candidate pose with the highest likelihood, and update the
  current inferred psoes to the candidate pose with highest likelihood if it
  increases the likelihood.

  Args:
    inferred_poses: Current inferred poses for all objects.
    candidate_poses: A set of candidate poses.
    compute_likelihood: A function evaluating the likelihood of different
      candidate poses.

  Returns:
    The updated inferred poses.
  """
  candidate_poses = jnp.concatenate(
      [inferred_poses[:, None], candidate_poses], axis=1
  )
  log_likelihoods = compute_likelihood(candidate_poses)
  inferred_poses = candidate_poses[:, jnp.argmax(log_likelihoods)]
  return inferred_poses


@dataclass
class InferenceProgram:
  """The inference program for 3DNEL MSIGP.

  Attributes:
    n_passes_pose_hypotheses: Number of times to to through pose hypotheses.
    n_passes_icp: Number of times to make ICP moves.
    n_passes_finetune: Number of times to do random walk finetuning.
    n_samples_per_iteration: Number of candidate poses to evaluate in parallel
      per iteration.
    var_concentration_list: List of variance and concentration parameters to use
      for sampling candidate poses.
    icp_var_concentration: Variance and concentration parameters for sampling
      candidate psoes around ICP results.
    use_flip: Whether to use flip proposals.
  """

  n_passes_pose_hypotheses: int = 1
  n_passes_icp: int = 1
  n_passes_finetune: int = 1
  n_samples_per_iteration: int = 80
  var_concentration_list: Sequence[Tuple[float, float]] = (
      (10.0, 300.0),
      (10.0, 800.0),
      (0.01, 800.0),
      (2.0, 2000.0),
  )
  icp_var_concentration: Tuple[float, float] = (10.0, 800.0)
  use_flip: bool = True

  def __post_init__(self):
    flip_transforms = []
    if self.use_flip:
      for dimension in range(3):
        euler = np.zeros(3)
        euler[dimension] = np.pi
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler('xyz', euler).as_matrix()
        flip_transforms.append(transform)

    flip_transforms.append(np.eye(4))
    self.flip_transforms = jax.device_put(flip_transforms)

  def infer(
      self,
      key: jnp.ndarray,
      initial_poses: np.ndarray,
      pose_hypotheses: Sequence[np.ndarray],
      compute_likelihood: Callable,
      icp: Optional[ICP] = None,
  ) -> jnp.ndarray:
    """Returns the inferred object poses using the inference program.

    Args:
      key: PRNGKey
      initial_poses: Array of shape (n_objs, 4, 4). The initial pose estimates.
      pose_hypotheses: Each element is an array of shape (n_hypotheses, 4, 4),
        and contains the pose hypotheses for a particular object.
      compute_likelihood: Function to evaluate the likelihood of given object
        poses.
      icp: Optional module implementing ICP-based updates.
    """
    n_objs = initial_poses.shape[0]
    assert len(pose_hypotheses) == n_objs
    inferred_poses = jax.device_put(initial_poses)
    pose_hypotheses = jax.device_put(pose_hypotheses)
    for _ in range(self.n_passes_pose_hypotheses):
      for obj_idx in np.random.permutation(n_objs):
        candidate_poses = (
            jnp.tile(
                inferred_poses[:, None],
                (1, pose_hypotheses[obj_idx].shape[0], 1, 1),
            )
            .at[obj_idx]
            .set(pose_hypotheses[obj_idx])
        )
        inferred_poses = move_to_candidate_poses(
            inferred_poses, candidate_poses, compute_likelihood
        )

    if icp is not None:
      var, concentration = self.icp_var_concentration
      for _ in range(self.n_passes_icp):
        for obj_idx in np.random.permutation(n_objs):
          for dimension in range(len(self.flip_transforms)):
            try:
              icp_pose = jax.device_put(
                  icp.fit(
                      inferred_poses[obj_idx].dot(
                          self.flip_transforms[dimension]
                      ),
                      obj_idx,
                  )
              )
            except ZeroDivisionError:
              print('ICP crashed. Using original object pose...')
              icp_pose = inferred_poses[obj_idx]

            key, candidate_poses = generate_candidate_poses(
                key,
                inferred_poses.at[obj_idx].set(icp_pose),
                obj_idx,
                var * jnp.ones(3),
                concentration,
                self.n_samples_per_iteration,
            )
            inferred_poses = move_to_candidate_poses(
                inferred_poses, candidate_poses, compute_likelihood
            )

    for _ in range(self.n_passes_finetune):
      for var, concentration in self.var_concentration_list:
        for obj_idx in np.random.permutation(n_objs):
          key, candidate_poses = generate_candidate_poses(
              key,
              inferred_poses,
              obj_idx,
              var * jnp.ones(3),
              concentration,
              self.n_samples_per_iteration,
          )
          inferred_poses = move_to_candidate_poses(
              inferred_poses, candidate_poses, compute_likelihood
          )

    return inferred_poses
