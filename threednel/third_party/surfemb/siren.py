# MIT License
#
# Copyright (c) 2020 Vincent Sitzmann
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# From https://vsitzmann.github.io/siren/ (MIT License)
import numpy as np
import torch
from torch import nn


class SineLayer(nn.Module):

  def __init__(
      self, in_features, out_features, bias=True, is_first=False, omega_0=30.0
  ):
    super().__init__()
    self.omega_0 = omega_0
    self.is_first = is_first
    self.in_features = in_features
    self.linear = nn.Linear(in_features, out_features, bias=bias)
    self.init_weights()

  def init_weights(self):
    with torch.no_grad():
      if self.is_first:
        self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
      else:
        self.linear.weight.uniform_(
            -np.sqrt(6 / self.in_features) / self.omega_0,
            np.sqrt(6 / self.in_features) / self.omega_0,
        )

  def forward(self, input):
    return torch.sin(self.omega_0 * self.linear(input))


class Siren(nn.Module):

  def __init__(
      self,
      in_features,
      hidden_features,
      hidden_layers,
      out_features,
      outermost_linear=True,
      first_omega_0=30.0,
      hidden_omega_0=30.0,
  ):
    super().__init__()
    self.net = []
    self.net.append(
        SineLayer(
            in_features, hidden_features, is_first=True, omega_0=first_omega_0
        )
    )
    for i in range(hidden_layers):
      self.net.append(
          SineLayer(
              hidden_features,
              hidden_features,
              is_first=False,
              omega_0=hidden_omega_0,
          )
      )
    if outermost_linear:
      final_linear = nn.Linear(hidden_features, out_features)
      with torch.no_grad():
        final_linear.weight.uniform_(
            -np.sqrt(6 / hidden_features) / hidden_omega_0,
            np.sqrt(6 / hidden_features) / hidden_omega_0,
        )
      self.net.append(final_linear)
    else:
      self.net.append(
          SineLayer(
              hidden_features,
              out_features,
              is_first=False,
              omega_0=hidden_omega_0,
          )
      )
    self.net = nn.Sequential(*self.net)

  def forward(self, coords):
    return self.net(coords)
