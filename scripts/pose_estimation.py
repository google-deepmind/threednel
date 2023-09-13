"""Script to reproduce the 6D object pose estimation results on YCB-V."""
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
import hashlib
import json
import os

import jax
import joblib
import numpy as np
import taichi as ti
from absl import app, flags
from threednel.bop.data import BOPTestDataset, RGBDImage
from threednel.bop.detector import Detector
from threednel.bop.results import generate_csv
from threednel.third_party.surfemb.utils import timer
from tqdm import tqdm

_EXPERIMENT_NAME = flags.DEFINE_string(
    'experiment_name', None, 'Name of the experiment.'
)
_SCALE_FACTOR = flags.DEFINE_float(
    'scale_factor', 0.25, 'Scale factor to run detection on.'
)
_FILL_IN_DEPTH = flags.DEFINE_boolean(
    'fill_in_depth', True, 'Whether to fill in missing depth.'
)
_N_PASSES_POSE_HYPOTHESES = flags.DEFINE_integer(
    'n_passes_pose_hypotheses',
    1,
    'Number of passes to propose to pose hypotheses.',
)
_N_PASSES_ICP = flags.DEFINE_integer(
    'n_passes_icp', 1, 'Number of passes to propose to ICP poses.'
)
_N_PASSES_FINETUNE = flags.DEFINE_integer(
    'n_passes_finetune', 1, 'Number of passes to do finetuning.'
)
_USE_CROPS = flags.DEFINE_boolean(
    'use_crops', True, 'Whether to use crop in generating pose hypotheses.'
)


def main(_):
  ti.init(arch=ti.cuda)
  experiment_name = _EXPERIMENT_NAME.value
  data_directory = os.environ['BOP_DATA_DIR']
  data = BOPTestDataset(
      data_directory=data_directory,
      load_detector_crops=True,
  )
  detector = Detector(
      data_directory=data_directory,
      n_passes_pose_hypotheses=_N_PASSES_POSE_HYPOTHESES.value,
      n_passes_icp=_N_PASSES_ICP.value,
      n_passes_finetune=_N_PASSES_FINETUNE.value,
      use_crops=_USE_CROPS.value,
  )
  flag_values = jax.tree_util.tree_map(
      lambda x: str(x),
      dict(
          experiment_name=_EXPERIMENT_NAME.value,
          scale_factor=_SCALE_FACTOR.value,
          fill_in_depth=_FILL_IN_DEPTH.value,
          n_passes_pose_hypotheses=_N_PASSES_POSE_HYPOTHESES.value,
          n_passes_icp=_N_PASSES_ICP.value,
          n_passes_finetune=_N_PASSES_FINETUNE.value,
          use_crops=_USE_CROPS.value,
      ),
  )
  results_hash = hashlib.md5(json.dumps(flag_values).encode('utf-8')).hexdigest()
  results_directory = os.path.join(
      data_directory,
      'results',
      f'pose_estimates_{experiment_name}_{results_hash}',
  )
  print(f'Working on results directory {results_directory}.')
  if not os.path.exists(results_directory):
    os.makedirs(results_directory)

  joblib.dump(flag_values, os.path.join(results_directory, 'flags.joblib'))
  with open(os.path.join(results_directory, 'flags.json'), 'w') as f:
    json.dump(flag_values, f)

  for scene_id in np.sort(np.unique(data.test_targets['scene_id'])):
    test_scene = data[scene_id]
    for img_id in test_scene.img_indices:
      print(f'Working on scene {scene_id}, image {img_id}.')
      results_fname = os.path.join(
          results_directory,
          f'scene_{scene_id}_img_{img_id}.joblib',
      )
      if os.path.exists(results_fname):
        continue

      bop_img = test_scene[img_id]
      test_img = RGBDImage(
          rgb=bop_img.rgb,
          depth=bop_img.depth,
          intrinsics=bop_img.intrinsics,
          bop_obj_indices=np.array(bop_img.bop_obj_indices),
          fill_in_depth=_FILL_IN_DEPTH.value,
          max_depth=1260.0,
          annotations=bop_img.annotations,
      )
      with timer(f'Inference for scene {scene_id}, image {img_id}'):
        detection_results = detector.detect(
            img=test_img,
            key=jax.random.PRNGKey(np.random.randint(0, 100000)),
            scale_factor=_SCALE_FACTOR.value,
        )

      joblib.dump(
          dict(
              bop_obj_indices=test_img.bop_obj_indices,
              gt_poses=bop_img.get_gt_poses(),
              initial_poses=detection_results.initial_poses,
              inferred_poses=detection_results.inferred_poses,
          ),
          results_fname,
      )

  predictions = []
  voting_predictions = []
  for scene_id in np.sort(np.unique(data.test_targets['scene_id'])):
    print(f'Working on scene {scene_id}.')
    test_scene = data[scene_id]
    for img_id in tqdm(test_scene.img_indices):
      results_fname = os.path.join(
          results_directory,
          f'scene_{scene_id}_img_{img_id}.joblib',
      )
      if not os.path.exists(results_fname):
        continue

      with open(results_fname, 'rb') as f:
        results = joblib.load(f)

      for obj_idx, bop_obj_idx in enumerate(results['bop_obj_indices']):
        predictions.append(
            dict(
                scene_id=scene_id,
                im_id=img_id,
                obj_id=bop_obj_idx,
                score=-1,
                R=results['inferred_poses'][obj_idx][:3, :3],
                t=results['inferred_poses'][obj_idx][:3, -1],
                time=-1,
            )
        )
        voting_predictions.append(
            dict(
                scene_id=scene_id,
                im_id=img_id,
                obj_id=bop_obj_idx,
                score=-1,
                R=results['initial_poses'][obj_idx][:3, :3],
                t=results['initial_poses'][obj_idx][:3, -1],
                time=-1,
            )
        )

  generate_csv(
      predictions,
      os.path.join(
          results_directory,
          (
              f'ycbv-threednel-{str(experiment_name).replace("_", "-")}-{results_hash}_ycbv-test.csv'
          ),
      ),
  )
  generate_csv(
      voting_predictions,
      os.path.join(
          results_directory,
          (
              f'ycbv-voting-{str(experiment_name).replace("_", "-")}-{results_hash}_ycbv-test.csv'
          ),
      ),
  )


if __name__ == '__main__':
  app.run(main)
