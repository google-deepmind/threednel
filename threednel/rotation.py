"""Module containing utils for discretizing the rotation space."""
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

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation as R


def fibonacci_sphere(n_points: int = 1000):
  """Evenly distribute points on a sphere using fibonacci sphere.

  https://extremelearning.com.au/evenly-distributing-points-on-a-sphere/

  Args:
    n_points: Number of samples on the sphere.

  Returns:
    n_points evenly distributed points on the sphere using fibonacci sphere.
  """
  phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle in radians
  y = 1 - (np.arange(n_points) / (n_points - 1)) * 2
  radius = np.sqrt(1 - y**2)
  theta = phi * np.arange(n_points)
  x = np.cos(theta) * radius
  z = np.sin(theta) * radius
  points = np.stack([x, y, z], axis=1)
  return points


@jax.jit
@functools.partial(jax.vmap, in_axes=(None, 0), out_axes=0)
def get_rotation_vectors(source: jnp.ndarray, target: jnp.ndarray):
  """Generate a rotation to rotate the source vector to the target vector.

  Args:
      source: Array of shape (3,) representing the source vector.
      target: Array of shape (3,) representing the target vector.

  Returns:
      Array of shape (3,), representing the rotation vector.
  """
  perp = jnp.cross(source, target)
  perp = perp / jnp.linalg.norm(perp)
  theta = jnp.arctan2(target @ jnp.cross(perp, source), target @ source)
  rotvec = theta * perp
  return rotvec


def generate_prototype_rotations(
    num_points_on_sphere: int = 200,
    num_inplane_rotations: int = 32,
    principal_axis: jnp.ndarray = jnp.array([0, 0, 1.0]),
) -> R:
  """Generate a set of prototype rotations to discretize the rotation space.

  Each prototype rotation first rotates the given principal axis to a direction
  specified by a point on the unit sphere, and then rotates the object around
  the resulting axis by an in-plane rotation.

  Args:
      num_points_on_sphere (int): num_points_on_sphere
      num_inplane_rotations (int): num_inplane_rotations
      principal_axis (jnp.ndarray): principal_axis

  Returns:
      R:
  """
  points_on_sphere = fibonacci_sphere(num_points_on_sphere)
  rotation_vectors = get_rotation_vectors(principal_axis, points_on_sphere)
  rotate_z = R.from_rotvec(rotation_vectors)
  prototype_rotations = []
  for ii in range(points_on_sphere.shape[0]):
    prototype_rotations.append(
        (
            R.from_rotvec(
                np.linspace(
                    0, 2 * np.pi, num_inplane_rotations, endpoint=False
                )[:, None]
                * points_on_sphere[ii]
            )
            * rotate_z[ii]
        ).as_quat()
    )
  prototype_rotations = R.from_quat(np.concatenate(prototype_rotations, axis=0))
  return prototype_rotations
