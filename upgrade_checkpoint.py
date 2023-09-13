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

from pytorch_lightning.utilities import migration
import torch

data_directory = os.environ['BOP_DATA_DIR']
checkpoint_path = os.path.join(
    data_directory, 'models', 'ycbv-jwpvdij1.compact.ckpt'
)
with migration.pl_legacy_patch():
  checkpoint = torch.load(checkpoint_path)

checkpoint['pytorch-lightning_version'] = '1.6.5'
migration.migrate_checkpoint(checkpoint)
torch.save(checkpoint, checkpoint_path)