"""Module containing distribution-related utilities."""
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
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp


def quaternion_to_rotation_matrix(Q: jnp.ndarray) -> jnp.ndarray:
  """Covert a quaternion into a full three-dimensional rotation matrix.

  Args:
    Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

  Returns:
    A 3x3 element matrix representing the full 3D rotation matrix.
    This rotation matrix converts a point in the local reference
    frame to a point in the global reference frame.
  """
  # Extract the values from Q
  q0 = Q[0]
  q1 = Q[1]
  q2 = Q[2]
  q3 = Q[3]

  # First row of the rotation matrix
  r00 = 2 * (q0 * q0 + q1 * q1) - 1
  r01 = 2 * (q1 * q2 - q0 * q3)
  r02 = 2 * (q1 * q3 + q0 * q2)

  # Second row of the rotation matrix
  r10 = 2 * (q1 * q2 + q0 * q3)
  r11 = 2 * (q0 * q0 + q2 * q2) - 1
  r12 = 2 * (q2 * q3 - q0 * q1)

  # Third row of the rotation matrix
  r20 = 2 * (q1 * q3 - q0 * q2)
  r21 = 2 * (q2 * q3 + q0 * q1)
  r22 = 2 * (q0 * q0 + q3 * q3) - 1

  # 3x3 rotation matrix
  rot_matrix = jnp.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])

  return rot_matrix


def gaussian_vmf(
    key: jnp.ndarray, var: jnp.ndarray, concentration: float
) -> jnp.ndarray:
  """Function to sample from the a GaussianVMF distribution."""
  translation = tfp.distributions.MultivariateNormalFullCovariance(
      loc=jnp.zeros(3), covariance_matrix=jnp.diag(var)
  ).sample(seed=key)
  quat = tfp.distributions.VonMisesFisher(
      jnp.array([1.0, 0.0, 0.0, 0.0]), concentration
  ).sample(seed=key)
  rot_matrix = quaternion_to_rotation_matrix(quat)
  return jnp.vstack([
      jnp.hstack([rot_matrix, translation.reshape(3, 1)]),
      jnp.array([0.0, 0.0, 0.0, 1.0]),
  ])


def gaussian_vmf_sample(
    key: jnp.ndarray,
    pose_mean: jnp.ndarray,
    var: jnp.ndarray,
    concentration: jnp.ndarray,
) -> jnp.ndarray:
  """Function to sample from a GaussianVMF distribution centered at pose_mean."""
  return pose_mean.dot(gaussian_vmf(key, var, concentration))


def gaussian_sample(
    key: jnp.ndarray, pose_mean: jnp.ndarray, var: jnp.ndarray
) -> jnp.ndarray:
  """Sample from a Gaissuain distribution centered at pose_mean."""
  translation = tfp.distributions.MultivariateNormalFullCovariance(
      loc=jnp.zeros(3), covariance_matrix=jnp.diag(var)
  ).sample(seed=key)
  return pose_mean.at[:3, -1].add(translation)
