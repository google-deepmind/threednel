"""Wrapper around https://github.com/nishadgothoskar/pararender.

Efficient parallel rendering of a large number of 3D scene descriptions.
"""
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
from typing import NamedTuple
import jax.numpy as jnp

class RenderedData(NamedTuple):
  """Container class holding the rendered results.

  Attributes:
    model_xyz: Coordinates of the rendered points in camera frame.
    obj_coords: Coordinates of the rendered points in object frame.
    obj_ids: Object ids of the rendered points.
  """

  model_xyz: jnp.ndarray
  obj_coords: jnp.ndarray
  obj_ids: jnp.ndarray
