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
from dataclasses import dataclass
from typing import NamedTuple, Sequence

import jax
import jax.dlpack
import jax.numpy as jnp
import numpy as np
import pararender.common as dr
from threednel.renderer import camera
import torch
import trimesh

RENDERER_ENV = None
PROJ_LIST = None


def setup_renderer(
    h: int,
    w: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    near: float,
    far: float,
    num_layers: int = 2048,
):
  """Function to set up a renderer.

  Args:
    h: Height of the image
    w: Width of the image
    fx: Focal length in the x direction
    fy: Focal length in the y direction
    cx: Principal point in the x direction
    cy: Principal point in the y direction
    near: Near plane distance
    far: Far plane distance
    num_layers: Number of layers used in rendering
  """
  global RENDERER_ENV
  global PROJ_LIST
  RENDERER_ENV = dr.RasterizeGLContext(h, w, output_db=False)
  PROJ_LIST = list(
      camera.open_gl_projection_matrix(h, w, fx, fy, cx, cy, near, far).reshape(
          -1
      )
  )
  dr._get_plugin(gl=True).setup(RENDERER_ENV.cpp_wrapper, h, w, num_layers)


def load_model(mesh: trimesh.Trimesh):
  """Function to load a object mesh model into the renderer."""
  vertices = np.array(mesh.vertices)
  vertices = np.concatenate(
      [vertices, np.ones((*vertices.shape[:-1], 1))], axis=-1
  )
  triangles = np.array(mesh.faces)
  dr._get_plugin(gl=True).load_vertices_fwd(
      RENDERER_ENV.cpp_wrapper,
      torch.tensor(vertices.astype("f"), device="cuda"),
      torch.tensor(triangles.astype(np.int32), device="cuda"),
  )


def render_to_torch(poses: jnp.ndarray, idx: int, on_object: int = 0):
  """Function to render a set of poses in parallel for a particular object."""
  poses_torch = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(poses))
  images_torch = dr._get_plugin(gl=True).rasterize_fwd_gl(
      RENDERER_ENV.cpp_wrapper, poses_torch, PROJ_LIST, idx, on_object
  )
  return images_torch


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


@dataclass
class ParallelRenderer:
  """Class for interacting with efficient parallel rendering.

  Attributes:
    height: Height of the image
    width: Width of the image
    intrinsics: Camera intrinsics matrix
    scale_factor: Scale factor at which to do the rendering.
  """

  height: int
  width: int
  intrinsics: np.ndarray
  scale_factor: float = 1.0
  near = 10.0
  far = 1500.0

  def __post_init__(self):
    fx, fy, cx, cy = (
        self.intrinsics[0, 0],
        self.intrinsics[1, 1],
        self.intrinsics[0, 2],
        self.intrinsics[1, 2],
    )
    self.mesh_names = []
    self.meshes = []
    self.offsets = jnp.zeros((1, 3))
    self.scales = jnp.ones(1)
    self.original_camera_params = (
        self.height,
        self.width,
        fx,
        fy,
        cx,
        cy,
        self.near,
        self.far,
    )
    height, width, fx, fy, cx, cy = camera.scale_camera_parameters(
        self.height, self.width, fx, fy, cx, cy, self.scale_factor
    )
    self.height, self.width = height, width
    self.camera_params = (height, width, fx, fy, cx, cy, self.near, self.far)
    setup_renderer(height, width, fx, fy, cx, cy, self.near, self.far)

  def add_trimesh(self, mesh: trimesh.Trimesh, mesh_name: str | None = None):
    """Function to add an object mesh to the renderer."""
    self.meshes.append(mesh)
    if mesh_name is None:
      num_objects = len(self.mesh_names)
      mesh_name = f"type_{num_objects}"

    self.mesh_names.append(mesh_name)
    bounding_sphere = mesh.bounding_sphere.primitive
    offset, scale = bounding_sphere.center, bounding_sphere.radius
    self.offsets = jnp.concatenate([self.offsets, offset[None]], axis=0)
    self.scales = jnp.concatenate([self.scales, jnp.array([scale])])
    load_model(mesh)

  def render(self, poses: np.ndarray, indices: Sequence[int]) -> RenderedData:
    """Function to render a set of object poses in parallel.

    Args:
        poses: Array of shape (n_objs, n_particles, 4, 4)
        indices: Sequence. Elements ranges from 0 to n_objs - 1

    Returns:
        Rendering results in RenderedData format.
    """
    images = jax.dlpack.from_dlpack(
        torch.utils.dlpack.to_dlpack(
            render_to_torch(poses, indices, on_object=0)
        )
    )
    model_xyz = images[..., :3]
    obj_ids = jnp.round(images[..., 3]).astype(jnp.int32) - 1
    offsets_array = self.offsets[obj_ids + 1]
    scales_array = self.scales[obj_ids + 1]
    obj_coords = jax.dlpack.from_dlpack(
        torch.utils.dlpack.to_dlpack(
            render_to_torch(poses, indices, on_object=1)
        )
    )[..., :3]
    obj_coords = (obj_coords - offsets_array) / scales_array[..., None]
    return RenderedData(
        model_xyz=model_xyz,
        obj_coords=obj_coords,
        obj_ids=obj_ids,
    )
