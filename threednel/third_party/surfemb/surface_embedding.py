# MIT License

# Copyright (c) 2022 Rasmus Laurvig Haugaard

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# Adapted from https://github.com/rasmushaugaard/surfemb/blob/master/surfemb/surface_embedding.py
# Updated object indices to range from 1 to 21 instead of from 0 to 20
# Deleted a few unused functions
from typing import Sequence, Union

import numpy as np
import pytorch_lightning as pl
import torch

from .siren import Siren
from .unet import ResNetUNet

mlp_class_dict = dict(siren=Siren)

imagenet_stats = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def tfms_normalize(img: np.ndarray):  # (h, w, 3) -> (3, h, w)
  mu, std = imagenet_stats
  if img.dtype == np.uint8:
    img = img / 255
  img = (img - mu) / std
  return img.transpose(2, 0, 1).astype(np.float32)


def tfms_denormalize(img: Union[np.ndarray, torch.Tensor]):
  mu, std = imagenet_stats
  if isinstance(img, torch.Tensor):
    mu, std = [
        torch.Tensor(v).type(img.dtype).to(img.device)[:, None, None]
        for v in (mu, std)
    ]
  return img * std + mu


class SurfaceEmbeddingModel(pl.LightningModule):

  def __init__(
      self,
      n_objs: int,
      emb_dim=12,
      n_pos=1024,
      n_neg=1024,
      lr_cnn=3e-4,
      lr_mlp=3e-5,
      mlp_name='siren',
      mlp_hidden_features=256,
      mlp_hidden_layers=2,
      key_noise=1e-3,
      warmup_steps=2000,
      separate_decoders=True,
      **kwargs,
  ):
    """:param emb_dim: number of embedding dimensions

    :param n_pos: number of positive (q, k) pairs from the object mask
    :param n_neg: number of negative keys, k-, from the object surface
    """
    super().__init__()
    self.save_hyperparameters()

    self.n_objs, self.emb_dim = n_objs, emb_dim
    self.n_pos, self.n_neg = n_pos, n_neg
    self.lr_cnn, self.lr_mlp = lr_cnn, lr_mlp
    self.warmup_steps = warmup_steps
    self.key_noise = key_noise
    self.separate_decoders = separate_decoders

    # query model
    self.cnn = ResNetUNet(
        n_class=(emb_dim + 1) if separate_decoders else n_objs * (emb_dim + 1),
        n_decoders=n_objs if separate_decoders else 1,
    )
    # key models
    mlp_class = mlp_class_dict[mlp_name]
    mlp_args = dict(
        in_features=3,
        out_features=emb_dim,
        hidden_features=mlp_hidden_features,
        hidden_layers=mlp_hidden_layers,
    )
    self.mlps = torch.nn.Sequential(
        *[mlp_class(**mlp_args) for _ in range(n_objs)]
    )
    self.bop_obj_indices_to_obj_indices = None

  def init_bop_obj_indices_to_obj_indices(self, bop_obj_indices: Sequence[int]):
    assert self.bop_obj_indices_to_obj_indices is None
    self.bop_obj_indices_to_obj_indices = {}
    for obj_idx, bop_obj_idx in enumerate(sorted(bop_obj_indices)):
      self.bop_obj_indices_to_obj_indices[bop_obj_idx] = obj_idx

  @torch.no_grad()
  def infer_cnn(
      self,
      img: Union[np.ndarray, torch.Tensor],
      bop_obj_idx: int,
      rotation_ensemble=True,
  ):
    assert not self.training
    assert self.bop_obj_indices_to_obj_indices is not None
    obj_idx = self.bop_obj_indices_to_obj_indices[bop_obj_idx]
    if isinstance(img, np.ndarray):
      if img.dtype == np.uint8:
        img = tfms_normalize(img)
      img = torch.from_numpy(img).to(self.device)
    _, h, w = img.shape

    if rotation_ensemble:
      img = utils.rotate_batch(img)  # (4, 3, h, h)
    else:
      img = img[None]  # (1, 3, h, w)
    cnn_out = self.cnn(
        img, [obj_idx] * len(img) if self.separate_decoders else None
    )
    if not self.separate_decoders:
      channel_idxs = [obj_idx] + list(
          self.n_objs + obj_idx * self.emb_dim + np.arange(self.emb_dim)
      )
      cnn_out = cnn_out[:, channel_idxs]
    # cnn_out: (B, 1+emb_dim, h, w)
    if rotation_ensemble:
      cnn_out = utils.rotate_batch_back(cnn_out).mean(dim=0)
    else:
      cnn_out = cnn_out[0]
    mask_lgts, query_img = cnn_out[0], cnn_out[1:]
    query_img = query_img.permute(1, 2, 0)  # (h, w, emb_dim)
    return mask_lgts, query_img

  @torch.no_grad()
  def infer_mlp(
      self, pts_norm: Union[np.ndarray, torch.Tensor], bop_obj_idx: int
  ):
    assert not self.training
    assert self.bop_obj_indices_to_obj_indices is not None
    obj_idx = self.bop_obj_indices_to_obj_indices[bop_obj_idx]
    if isinstance(pts_norm, np.ndarray):
      pts_norm = torch.from_numpy(pts_norm).to(self.device).float()
    return self.mlps[obj_idx](pts_norm)  # (..., emb_dim)

  def get_emb_vis(
      self,
      emb_img: torch.Tensor,
      mask: torch.Tensor = None,
      demean: torch.tensor = False,
  ):
    if demean is True:
      demean = emb_img[mask].view(-1, self.emb_dim).mean(dim=0)
    if demean is not False:
      emb_img = emb_img - demean
    shape = emb_img.shape[:-1]
    emb_img = emb_img.view(*shape, 3, -1).mean(dim=-1)
    if mask is not None:
      emb_img[~mask] = 0.0
    emb_img /= torch.abs(emb_img).max() + 1e-9
    emb_img.mul_(0.5).add_(0.5)
    return emb_img
