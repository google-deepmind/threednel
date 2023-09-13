"""Module containing classes for detector and detection results."""
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
import os
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import jax
import jax.dlpack
import jax.numpy as jnp
import numpy as np
import torch
from threednel import ndl
from threednel.bop.bop_surfemb import BOPSurfEmb
from threednel.bop.bop_vote import BOPVoting
from threednel.bop.data import RGBDImage
from threednel.bop.hypotheses import HypothesesGeneration
from threednel.icp import ICP
from threednel.inference_program import InferenceProgram
from threednel.renderer.parallel import ParallelRenderer


@dataclass
class DetectionResults:
  """Container class for detection results.

  Attributes:
    initial_poses: Initial pose estimation.
    inferred_poses: Final inferred pose estimation.
    renderer: Constructed renderer.
    query_embeddings: Relevant query embeddings.
    data_mask: Data masks.
    pose_hypotheses: All the generated pose hypotheses.
  """

  initial_poses: jnp.ndarray
  inferred_poses: jnp.ndarray
  renderer: ParallelRenderer
  query_embeddings: jnp.ndarray
  data_mask: jnp.ndarray
  pose_hypotheses: Sequence[jnp.ndarray]


@dataclass
class Detector:
  """A detector implementing 6D pose estimation using 3DNEL."""

  data_directory: str = os.environ['BOP_DATA_DIR']
  device: str = 'cuda:0'
  n_objs_range: Tuple[int, int] = (3, 7)
  # Params for hypotheses generation
  mask_threshold: float = 0.7
  n_top_translations: int = 20
  n_pose_hypotheses_per_crop: int = 80
  maximum_filter_size: int = 10
  n_top_rotations_per_translation: int = 10
  n_pose_hypotheses_per_object: int = 30
  default_scale: float = 1.5
  use_crops: bool = True
  # Params for likelihood
  likelihood_factory: Callable = ndl.JAXNDL
  r: float = 5.0
  outlier_prob: float = 0.01
  outlier_volume: float = 1000.0**3
  outlier_scaling: float = 1 / 70000
  filter_shape: Tuple[int, int] = (10, 10)
  # Params for inference program
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
  use_flip: bool = True
  # Params for ICP
  icp_var_concentration: Tuple[float, float] = (10.0, 800.0)
  icp_n_outer_iterations: int = 5
  icp_n_inner_iterations: int = 2

  def __post_init__(self):
    self.bop_surfemb = BOPSurfEmb(
        surfemb_model_path=os.path.join(
            self.data_directory, 'models', 'ycbv-jwpvdij1.compact.ckpt'
        ),
        device=self.device,
    )
    self.bop_vote = BOPVoting(data_directory=self.data_directory, device=self.device)
    self.bop_vote.load_all_key_embeddings(self.bop_surfemb.surfemb_model)
    self.hypotheses_generation = HypothesesGeneration(
        bop_surfemb=self.bop_surfemb,
        bop_vote=self.bop_vote,
        mask_threshold=self.mask_threshold,
        n_top_translations=self.n_top_translations,
        n_pose_hypotheses_per_crop=self.n_pose_hypotheses_per_crop,
        maximum_filter_size=self.maximum_filter_size,
        n_top_rotations_per_translation=self.n_top_rotations_per_translation,
        n_pose_hypotheses_per_object=self.n_pose_hypotheses_per_object,
        default_scale=self.default_scale,
        use_crops=self.use_crops,
    )
    self.likelihood_dict = {}
    for n_objs in range(self.n_objs_range[0], self.n_objs_range[1]):
      self.likelihood_dict[n_objs] = self.likelihood_factory(
          model=self.bop_surfemb.surfemb_model,
          n_objs=n_objs,
          r=self.r,
          outlier_prob=self.outlier_prob,
          outlier_volume=self.outlier_volume,
          outlier_scaling=self.outlier_scaling,
          filter_shape=self.filter_shape,
          data_directory=self.data_directory,
      )

    self.inference_program = InferenceProgram(
        n_passes_pose_hypotheses=self.n_passes_pose_hypotheses,
        n_passes_icp=self.n_passes_icp,
        n_passes_finetune=self.n_passes_finetune,
        n_samples_per_iteration=self.n_samples_per_iteration,
        var_concentration_list=self.var_concentration_list,
        use_flip=self.use_flip,
    )

  def detect(
      self,
      img: RGBDImage,
      key: jnp.ndarray,
      scale_factor: float = 0.25,
      initial_poses: Optional[jnp.ndarray] = None,
  ):
    """Function to do pose estimation on a given RGBDImage.

    Args:
      img: The input RGBDImage.
      key: JAX PRNGKey.
      scale_factor: Scale factor used to scale the input RGBDImage.
      initial_poses: Initial poses. If none, use the top scoring pose hypothesis
        for each object as the initial poses.

    Returns:
      Detection results on the input RGBDImage.
    """
    (
        query_embeddings,
        data_mask,
        pose_hypotheses,
    ) = self.hypotheses_generation.generate(img)
    log_normalizers = self.bop_vote.get_log_normalizers(
        torch.from_dlpack(jax.dlpack.to_dlpack(query_embeddings)),
        img.bop_obj_indices,
        squeeze=False,
    )
    n_objs = len(img.bop_obj_indices)
    likelihood = self.likelihood_dict[n_objs]
    likelihood.set_for_new_img(
        img=img,
        query_embeddings=query_embeddings,
        log_normalizers=jax.dlpack.from_dlpack(torch.to_dlpack(log_normalizers)),
        data_mask=data_mask,
        scale_factor=scale_factor,
    )

    def render(pose: np.ndarray, obj_idx: int):
      """Function to render an objectin a given pose and get the point cloud and object mask.

      Args:
          pose: Array of shape (4, 4). Pose of the object.
          obj_idx: object index in the scene.

      Returns:
          cloud: Array of shape (n_points, 3). Rendered point cloud of the
          object.
      """
      pose = jax.device_put(pose)
      rendered_data = likelihood.renderer.render(pose[None, None], [obj_idx])
      model_mask = np.array(rendered_data.obj_ids[0] >= 0)
      cloud = np.array(rendered_data.model_xyz[0])[model_mask]
      return cloud, model_mask

    self.icp = ICP(
        target_cloud=likelihood.data_xyz[likelihood.data_mask],
        render=render,
        n_outer_iterations=self.icp_n_outer_iterations,
        n_inner_iterations=self.icp_n_inner_iterations,
    )
    if initial_poses is None:
      initial_poses = jnp.array([pose_hypotheses[ii][0] for ii in range(n_objs)])

    inferred_poses = self.inference_program.infer(
        key,
        initial_poses,
        pose_hypotheses,
        compute_likelihood=likelihood.compute_likelihood,
        icp=self.icp,
    )
    return DetectionResults(
        initial_poses=initial_poses,
        inferred_poses=inferred_poses,
        renderer=likelihood.renderer,
        query_embeddings=query_embeddings,
        data_mask=data_mask,
        pose_hypotheses=pose_hypotheses,
    )
