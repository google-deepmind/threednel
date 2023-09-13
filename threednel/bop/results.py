"""Module for converting results into the BOP challenge format."""
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
from typing import Any, Mapping, Sequence

import jax
import pandas as pd


def generate_csv(
    predictions: Sequence[Mapping[str, Any]], csv_filename: str
) -> pd.DataFrame:
  """Helper function to convert results into the BOP challenge format.

  Args:
      predictions: A sequence of mappings Each mapping should contain scene_id,
        im_id, obj_id, score, R, t and time
      csv_filename: Name of the csv file containing the pose estimation results
        in the BOP challenge format.

  Returns:
      The pandas dataframe containing the pose estimation results.
  """
  for prediction in predictions:
    prediction['R'] = ' '.join([str(ele) for ele in prediction['R'].ravel()])
    prediction['t'] = ' '.join([str(ele) for ele in prediction['t']])

  outer_treedef = jax.tree_util.tree_structure([0] * len(predictions))
  inner_treedef = jax.tree_util.tree_structure(predictions[0])
  transposed_predictions = jax.tree_util.tree_transpose(
      outer_treedef, inner_treedef, predictions
  )
  columns = ['scene_id', 'im_id', 'obj_id', 'score', 'R', 't', 'time']
  df = pd.DataFrame(data=transposed_predictions)[columns]
  df.to_csv(csv_filename, index=False, header=False)
  return df
