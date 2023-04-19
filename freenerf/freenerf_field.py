# Copyright 2022 The Nerfstudio Team. All rights reserved.
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

"""
Field for compound nerf model, adds scene contraction and image embeddings to instant ngp
"""


from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torchtyping import TensorType

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import Encoding, HashEncoding, SHEncoding
from nerfstudio.field_components.field_heads import (
    DensityFieldHead,
    FieldHead,
    FieldHeadNames,
    PredNormalsFieldHead,
    RGBFieldHead,
    SemanticFieldHead,
    TransientDensityFieldHead,
    TransientRGBFieldHead,
    UncertaintyFieldHead,
)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import (
    SceneContraction,
    SpatialDistortion,
)
from nerfstudio.fields.base_field import Field, shift_directions_for_tcnn

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass


class TorchFreeNerfactoField(Field):
    """
    PyTorch implementation of the compound field.
    """

    def __init__(
        self,
        aabb: TensorType,
        num_images: int,
        position_encoding: Encoding = HashEncoding(),
        direction_encoding: Encoding = SHEncoding(),
        base_mlp_num_layers: int = 3,
        base_mlp_layer_width: int = 64,
        head_mlp_num_layers: int = 2,
        head_mlp_layer_width: int = 32,
        appearance_embedding_dim: int = 40,
        skip_connections: Tuple = (4,),
        field_heads: Tuple[FieldHead] = (RGBFieldHead(),),
        spatial_distortion: SpatialDistortion = SceneContraction(),
    ) -> None:
        super().__init__()
        self.aabb = Parameter(aabb, requires_grad=False)
        self.spatial_distortion = spatial_distortion
        self.num_images = num_images
        self.appearance_embedding_dim = appearance_embedding_dim
        self.embedding_appearance = Embedding(self.num_images, self.appearance_embedding_dim)

        self.position_encoding = position_encoding
        self.direction_encoding = direction_encoding

        self.mlp_base = MLP(
            in_dim=self.position_encoding.get_out_dim(),
            num_layers=base_mlp_num_layers,
            layer_width=base_mlp_layer_width,
            skip_connections=skip_connections,
            out_activation=nn.ReLU(),
        )

        self.mlp_head = MLP(
            in_dim=self.mlp_base.get_out_dim() + self.direction_encoding.get_out_dim() + self.appearance_embedding_dim,
            num_layers=head_mlp_num_layers,
            layer_width=head_mlp_layer_width,
            out_activation=nn.ReLU(),
        )

        self.field_output_density = DensityFieldHead(in_dim=self.mlp_base.get_out_dim())
        self.field_heads = nn.ModuleList(field_heads)
        for field_head in self.field_heads:
            field_head.set_in_dim(self.mlp_head.get_out_dim())  # type: ignore

    def get_density(self, ray_samples: RaySamples,freq_mask=None) -> Tuple[TensorType, TensorType]:
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
        else:
            positions = ray_samples.frustums.get_positions()
        if freq_mask is not None :
            (positions_freq_mask,dir_freq_mask) = freq_mask
            
            encoded_xyz = self.position_encoding(positions.view(-1, 3))*positions_freq_mask.view(1,-1)
        else:
            encoded_xyz = self.position_encoding(positions.view(-1, 3))
        base_mlp_out = self.mlp_base(encoded_xyz).view(*ray_samples.frustums.shape, -1)
        density = self.field_output_density(base_mlp_out)
        return density, base_mlp_out

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None,freq_mask=None
    ) -> Dict[FieldHeadNames, TensorType]:
        outputs_shape = ray_samples.frustums.directions.shape[:-1]
        
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        camera_indices = ray_samples.camera_indices.squeeze()
        if self.training:
            embedded_appearance = self.embedding_appearance(camera_indices)
        else:
            embedded_appearance = torch.zeros(
                (*outputs_shape, self.appearance_embedding_dim),
                device=ray_samples.frustums.directions.device,
            )

        outputs = {}
        if freq_mask is not None :
            (positions_freq_mask,dir_freq_mask) = freq_mask
        for field_head in self.field_heads:
            # encoded_dir = self.direction_encoding(ray_samples.frustums.directions.reshape(-1, 3)).view(
            #     *outputs_shape, -1
            # )
            if freq_mask is not None :
                encoded_dir = self.direction_encoding(ray_samples.frustums.directions.reshape(-1, 3))*dir_freq_mask.view(1,-1)
            else:
                encoded_dir = self.direction_encoding(ray_samples.frustums.directions.reshape(-1, 3))
            mlp_out = self.mlp_head(
                torch.cat(
                    [
                        encoded_dir,
                        density_embedding.view(-1, density_embedding.shape[-1]),  # type:ignore
                        embedded_appearance.view(-1, self.appearance_embedding_dim),
                    ],
                    dim=-1,  # type:ignore
                )
            ).view(*outputs_shape, -1)
            outputs[field_head.field_head_name] = field_head(mlp_out)
        return outputs
    def forward(self, ray_samples: RaySamples, compute_normals: bool = False,freq_mask=None) -> Dict[FieldHeadNames, TensorType]:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        if compute_normals:
            with torch.enable_grad():
                density, density_embedding = self.get_density(ray_samples,freq_mask=freq_mask)
        else:
            density, density_embedding = self.get_density(ray_samples,freq_mask=freq_mask)

        field_outputs = self.get_outputs(ray_samples, density_embedding=density_embedding,freq_mask=freq_mask)
        field_outputs[FieldHeadNames.DENSITY] = density  # type: ignore

        if compute_normals:
            with torch.enable_grad():
                normals = self.get_normals()
            field_outputs[FieldHeadNames.NORMALS] = normals  # type: ignore
        return field_outputs


field_implementation_to_class: Dict[str, Field] = {"torch": TorchFreeNerfactoField}
