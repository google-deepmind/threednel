"""Module containing useful classes for interacting with data in BOP format."""
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

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import imageio
import jax
import numpy as np
import tensorflow as tf
import trimesh
from threednel.renderer.camera import CameraIntrinsics
from threednel.renderer.parallel import ParallelRenderer
from threednel.third_party.ip_basic import fill_in_multiscale
from threednel.third_party.surfemb.utils import get_bbox_rgb_crop_K_crop


def depth_to_coords_in_camera(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    mask: Optional[np.ndarray] = None,
    as_image_shape: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
  """Convert depth image to coords in camera space for points in mask.

  Args:
      depth: Array of shape (H, W).
      intrinsics: Array of shape (3, 3), camera intrinsic matrix.
      mask: Array of shape (H, W), with 1s where points are quried.
      as_image_shape: If True, return arrays of shape (H, W, 3)

  Returns:
      np.ndarray: Array of shape (N, 3) or (H, W, 3), coordinates in camera
      space.
      np.ndarray: Array of shape (N, 2) or (H, W, 2), coordinates on image
      plane.
          N is the number of 1s in mask.
  """
  if as_image_shape:
    assert mask is None
    vu = np.mgrid[: depth.shape[0], : depth.shape[1]]
  else:
    if mask is None:
      mask = np.ones_like(depth)
    else:
      assert not as_image_shape
      assert mask.shape == depth.shape

    vu = np.nonzero(mask)

  depth_for_uv = depth[vu[0], vu[1]]
  full_vec = np.stack(
      [vu[1] * depth_for_uv, vu[0] * depth_for_uv, depth_for_uv], axis=0
  )
  coords_in_camera = np.moveaxis(
      np.einsum('ij,j...->i...', np.linalg.inv(intrinsics), full_vec), 0, -1
  )
  coords_on_image = np.moveaxis(vu, 0, -1)
  return coords_in_camera, coords_on_image


@dataclass
class RGBDImage:
  """Generic container class for an RGB-D image.

  Attributes:
    rgb: The RGB image.
    depth: The depth map in milimeters.
    intrinsics: The 3x3 camera intrinsics matrix.
    bop_obj_indices: Array containing the indices of the objects in the scene
    fill_in_depth: Whether we fill in missing depth values.
    max_depth: Maximum depth used to fill in missing depth values.
    annotations: Optional annotations for each object in the scene.
  """

  rgb: np.ndarray
  depth: np.ndarray
  intrinsics: np.ndarray
  bop_obj_indices: np.ndarray
  fill_in_depth: bool = False
  max_depth: float = np.inf
  annotations: Optional[Sequence] = None

  def __post_init__(self):
    self.depth[self.depth > self.max_depth] = 0.0
    if self.fill_in_depth:
      assert self.max_depth < np.inf
      self.depth, _ = fill_in_multiscale(self.depth, max_depth=self.max_depth)

    if self.annotations is None:
      self.annotations = [{} for _ in self.bop_obj_indices]

  def unproject(self) -> np.ndarray:
    """Unproject pixels in the RGB-D image into 3D space (in camera frame)."""
    data_xyz, _ = depth_to_coords_in_camera(
        self.depth, self.intrinsics, as_image_shape=True
    )
    return data_xyz

  def scale(self, scale_factor: float) -> RGBDImage:
    """Scale the RGB-D image by the given scale factor."""
    camera_intrinsics = CameraIntrinsics.from_matrix(
        shape=self.depth.shape, intrinsics=self.intrinsics
    ).scale(scale_factor)
    shape = (camera_intrinsics.height, camera_intrinsics.width)
    return RGBDImage(
        rgb=np.round(
            jax.image.resize(self.rgb, shape=shape + (3,), method='bilinear')
        ).astype(int),
        depth=np.array(jax.image.resize(self.depth, shape=shape, method='nearest')),
        intrinsics=camera_intrinsics.intrinsics_matrix,
        bop_obj_indices=self.bop_obj_indices,
        fill_in_depth=self.fill_in_depth,
        max_depth=self.max_depth,
    )

  def get_renderer(
      self, data_directory: str, scale_factor: float = 1.0
  ) -> ParallelRenderer:
    """Construct a renderer that can render a set of 3D scene descriptions in parallel."""
    height, width = self.depth.shape
    gl_renderer = ParallelRenderer(
        height=height,
        width=width,
        intrinsics=self.intrinsics,
        scale_factor=scale_factor,
    )
    for bop_obj_idx in self.bop_obj_indices:
      gl_renderer.add_trimesh(
          trimesh.load(
              os.path.join(data_directory, f'bop/ycbv/models/obj_{bop_obj_idx:06d}.ply')
          ),
          mesh_name=bop_obj_idx,
      )

    return gl_renderer


@dataclass
class BOPTestImage:
  """Class for interacting with test images from the BOP dataset.

  Attributes:
      dataset: Name of the dataset.
      scene_id: ID of the scene in the BOP dataset.
      img_id: ID of the image for the scene in the BOP dataset.
      rgb: Array of shape (H, W, 3).
      depth: Array of shape (H, W). In milimeters.
      intrinsics: Array of shape (3, 3). Camera intrinsics matrix.
      camera_pose: Array of shape (4, 4). Transform matrix from world frame to
        camera frame.
      bop_obj_indices: BOP indices of the objects in the image. Ranges from 1 to
        21.
      annotations: Annotations for each object in the scene, including object
        pose information and optionally 2D detection results.
      default_scales: default scales at which we obtain query embeddings for
        each object.
  """

  dataset: str
  scene_id: int
  img_id: int
  rgb: np.ndarray
  depth: np.ndarray
  intrinsics: np.ndarray
  camera_pose: np.ndarray
  bop_obj_indices: tuple[int, ...]
  annotations: Sequence
  default_scales: np.ndarray

  def __post_init__(self):
    self.obj_id, self.inst_count = np.unique(self.bop_obj_indices, return_counts=True)
    assert np.all(np.repeat(self.obj_id, self.inst_count) == self.bop_obj_indices)

  def get_gt_poses(self) -> List[np.ndarray]:
    """Function to get ground-truth object poses for the objects in the scene."""
    gt_poses = [annotation['model_to_cam'].copy() for annotation in self.annotations]
    return gt_poses


@dataclass
class BOPTestScene:
  """Class for interacting with scenes from the BOP dataset."""

  scene_path: str
  load_detector_crops: bool = False

  def __post_init__(self):
    self.data_directory = str(Path(self.scene_path).parents[3])
    if self.data_directory.startswith('gs:/'):
      self.data_directory = self.data_directory[:4] + '/' + self.data_directory[4:]

    self.scene_id = int(os.path.basename(self.scene_path))
    with open(os.path.join(self.scene_path, 'scene_camera.json'), 'r') as f:
      self.scene_camera = json.load(f)

    with open(os.path.join(self.scene_path, 'scene_gt.json'), 'r') as f:
      self.scene_gt = json.load(f)

    with open(os.path.join(self.scene_path, 'scene_gt_info.json'), 'r') as f:
      self.scene_gt_info = json.load(f)

    self.img_indices = [int(img_id) for img_id in self.scene_camera.keys()]
    with open(
        os.path.join(
            self.data_directory,
            'bop',
            'ycbv',
            'camera_uw.json',
        ),
        'r',
    ) as f:
      self.camera_info = json.load(f)

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
    self.default_scales = {
        bop_obj_idx: (1.5 if self.models_info[bop_obj_idx]['diameter'] > 70 else 3.0)
        for bop_obj_idx in self.models_info
    }
    if self.load_detector_crops:
      detection_folder = os.path.join(self.data_directory, 'detection_results', 'ycbv')
      with tf.io.gfile.GFile(os.path.join(detection_folder, 'bboxes.npy'), 'rb') as f:
        self.bboxes = np.load(f)

      with tf.io.gfile.GFile(os.path.join(detection_folder, 'obj_ids.npy'), 'rb') as f:
        self.obj_ids = np.load(f)

      with tf.io.gfile.GFile(
          os.path.join(detection_folder, 'scene_ids.npy'), 'rb'
      ) as f:
        self.scene_ids = np.load(f)

      with tf.io.gfile.GFile(os.path.join(detection_folder, 'view_ids.npy'), 'rb') as f:
        self.view_ids = np.load(f)

  def __getitem__(self, img_id):
    img_id = str(img_id)
    img_fname = f'{int(img_id):06d}.png'
    rgb = imageio.imread(os.path.join(self.scene_path, 'rgb', img_fname))
    depth = (
        imageio.imread(os.path.join(self.scene_path, 'depth', img_fname))
        * self.camera_info['depth_scale']
    )
    intrinsics = np.array(self.scene_camera[img_id]['cam_K']).reshape((3, 3))

    cam_pose = np.eye(4)
    cam_pose[:3, :3] = np.array(
        self.scene_camera[img_id].get('cam_R_w2c', np.eye(3).ravel())
    ).reshape((3, 3))
    cam_pose[:3, -1] = np.array(self.scene_camera[img_id].get('cam_t_w2c', np.zeros(3)))

    annotations = []
    bop_obj_indices = []
    for instance_idx, instance in enumerate(self.scene_gt[img_id]):
      model_to_cam = np.eye(4)
      model_to_cam[:3, :3] = np.array(instance['cam_R_m2c']).reshape((3, 3))
      model_to_cam[:3, -1] = np.array(instance['cam_t_m2c'])
      mask = imageio.imread(
          os.path.join(
              self.scene_path,
              'mask',
              f'{int(img_id):06d}_{instance_idx:06d}.png',
          )
      )
      mask_visible = imageio.imread(
          os.path.join(
              self.scene_path,
              'mask_visib',
              f'{int(img_id):06d}_{instance_idx:06d}.png',
          )
      )
      annotation = {
          'model_to_cam': model_to_cam,
          'mask': mask,
          'mask_visible': mask_visible,
      }
      annotation.update(self.scene_gt_info[img_id][instance_idx])
      bop_obj_indices.append(instance['obj_id'])
      if self.load_detector_crops:
        detection_crop_idx = np.argwhere(
            (self.scene_ids == self.scene_id)
            * (self.view_ids == int(img_id))
            * (self.obj_ids == instance['obj_id'])
        )
        detector_crops = []
        for detection_crop_idx in detection_crop_idx.ravel():
          bbox = self.bboxes[detection_crop_idx]
          (
              crop_bbox_in_original,
              rgb_crop,
              depth_crop,
              K_crop,
          ) = get_bbox_rgb_crop_K_crop(rgb, depth, bbox, K=intrinsics)
          left, top, right, bottom = crop_bbox_in_original
          detector_crops.append(
              dict(
                  AABB_crop=crop_bbox_in_original,
                  rgb_crop=rgb_crop,
                  depth_crop=depth_crop,
                  K_crop=K_crop,
              )
          )

        annotation['detector_crops'] = detector_crops

      annotations.append(annotation)

    return BOPTestImage(
        dataset='ycbv',
        scene_id=self.scene_id,
        img_id=int(img_id),
        rgb=rgb,
        depth=depth,
        intrinsics=intrinsics,
        camera_pose=cam_pose,
        bop_obj_indices=tuple(bop_obj_indices),
        annotations=annotations,
        default_scales=np.array(
            [self.default_scales[bop_obj_idx] for bop_obj_idx in bop_obj_indices]
        ),
    )

  @property
  def images(self):
    for img_id in self.img_indices:
      yield self.__getitem__(img_id)


@dataclass
class BOPTestDataset:
  """Class for interacting with a BOP dataset."""

  data_directory: str
  load_detector_crops: bool = False

  def __post_init__(self):
    with open(
        os.path.join(self.data_directory, 'bop', 'ycbv', 'test_targets_bop19.json'),
        'r',
    ) as f:
      test_targets = json.load(f)

    outer_treedef = jax.tree_util.tree_structure([0] * len(test_targets))
    inner_treedef = jax.tree_util.tree_structure(test_targets[0])
    test_targets = jax.tree_util.tree_transpose(
        outer_treedef, inner_treedef, test_targets
    )
    self.test_targets = {key: np.array(test_targets[key]) for key in test_targets}
    with open(
        os.path.join(
            self.data_directory,
            'bop',
            'ycbv',
            'camera_uw.json',
        ),
        'r',
    ) as f:
      self.camera_info = json.load(f)

  def __getitem__(self, scene_id: int):
    assert np.sum(self.test_targets['scene_id'] == scene_id) > 0
    scene_path = os.path.join(
        self.data_directory,
        f'bop/ycbv/test/{scene_id:06d}',
    )
    return BOPTestScene(scene_path, load_detector_crops=self.load_detector_crops)
