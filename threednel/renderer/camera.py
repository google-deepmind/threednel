"""Module containing camera-related utilities."""
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
from __future__ import annotations
from typing import NamedTuple
import numpy as np


class CameraIntrinsics(NamedTuple):
  """Container class for camera intrinsics matrix.

  Attributes:
    height: Height of the image
    width: Width of the image
    fx: Focal length in the x direction
    fy: Focal length in the y direction
    cx: Principal point in the x direction
    cy: Principal point in the y direction
    near: Near plane distance
    far: Far plane distance
  """

  height: int
  width: int
  fx: float
  fy: float
  cx: float
  cy: float
  near: float = 10.0
  far: float = 10000.0

  @staticmethod
  def from_matrix(
      shape: tuple[int, int], intrinsics: np.ndarray
  ) -> CameraIntrinsics:
    """Construct a CameraIntrinsics object from a camera intrinsics matrix."""
    return CameraIntrinsics(
        height=shape[0],
        width=shape[1],
        fx=intrinsics[0, 0],
        fy=intrinsics[1, 1],
        cx=intrinsics[0, 2],
        cy=intrinsics[1, 2],
    )

  @property
  def intrinsics_matrix(self) -> np.ndarray:
    """Returns the 3x3 camera intrinsics matrix."""
    intrinsics = np.array(
        [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]]
    )
    return intrinsics

  def scale(self, scale_factor: float) -> CameraIntrinsics:
    """Returns the scaled CameraIntrinsics object."""
    return CameraIntrinsics(
        height=int(np.round(self.height * scale_factor)),
        width=int(np.round(self.width * scale_factor)),
        fx=self.fx * scale_factor,
        fy=self.fy * scale_factor,
        cx=self.cx * scale_factor,
        cy=self.cy * scale_factor,
        near=self.near,
        far=self.far,
    )


def open_gl_projection_matrix(
    h: int,
    w: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    near: float,
    far: float,
) -> np.ndarray:
  """Function to create OpenGL projection matrix.

  Args:
    h: Height of the image
    w: Width of the image
    fx: Focal length in the x direction
    fy: Focal length in the y direction
    cx: Principal point in the x direction
    cy: Principal point in the y direction
    near: Near plane distance
    far: Far plane distance

  Returns:
    OpenGL projection matrix.
  """
  # transform from cv2 camera coordinates to opengl (flipping sign of y and z)
  view = np.eye(4)
  view[1:3] *= -1

  # see http://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/
  persp = np.zeros((4, 4))
  persp[0, 0] = fx
  persp[1, 1] = fy
  persp[0, 2] = cx
  persp[1, 2] = cy
  persp[2, 2] = near + far
  persp[2, 3] = near * far
  persp[3, 2] = -1
  # transform the camera matrix from cv2 to opengl as well (flipping sign of
  # y and z)
  persp[:2, 1:3] *= -1

  # The origin of the image is in the *center* of the top left pixel.
  # The orthographic matrix should map the whole image *area* into the opengl
  # NDC, therefore the -.5 below:

  left, right, bottom, top = -0.5, w - 0.5, -0.5, h - 0.5
  orth = np.array([
      (2 / (right - left), 0, 0, -(right + left) / (right - left)),
      (0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)),
      (0, 0, -2 / (far - near), -(far + near) / (far - near)),
      (0, 0, 0, 1),
  ])
  return orth @ persp @ view


def scale_camera_parameters(
    h: int,
    w: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    scaling_factor: float,
):
  """Function to scale camera parameters.

  Args:
    h: Height of the image
    w: Width of the image
    fx: Focal length in the x direction
    fy: Focal length in the y direction
    cx: Principal point in the x direction
    cy: Principal point in the y direction
    scaling_factor: The scaling factor we use to scale the camera parameters.

  Returns:
    Scaled camera parameters.
  """
  new_fx = fx * scaling_factor
  new_fy = fy * scaling_factor
  new_cx = cx * scaling_factor
  new_cy = cy * scaling_factor

  new_h = int(np.round(h * scaling_factor))
  new_w = int(np.round(w * scaling_factor))
  return new_h, new_w, new_fx, new_fy, new_cx, new_cy
