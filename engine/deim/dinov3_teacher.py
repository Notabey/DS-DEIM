"""
RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models
Copyright (c) 2025 The RT-DETRv4 Authors. All Rights Reserved.

DINOv3 Teacher Model using HuggingFace Transformers
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core import register
import logging
import torchvision.transforms.v2 as transforms

_logger = logging.getLogger(__name__)


@register()
class DINOv3TeacherModel(nn.Module):
    """
    DINOv3 Teacher Model using HuggingFace Transformers library.
    Loads pre-trained DINOv3 (DINOv2 with registers) for feature distillation.
    """
    def __init__(self,
                 dinov3_weights_path: str,
                 dinov3_model_type: str = "dinov3_vitb16",
                 patch_size: int = 16,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        super().__init__()
        self.patch_size = patch_size

        _logger.info(f"[Teacher Model] Loading DINOv3 from HuggingFace: {dinov3_weights_path}")
        
        try:
            from transformers import AutoModel, AutoConfig
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")
        
        # Load model from HuggingFace format (local directory or hub)
        # The weights path should be a directory containing config.json and model.safetensors
        self.model = AutoModel.from_pretrained(
            dinov3_weights_path,
            trust_remote_code=True,
            local_files_only=True  # Use local files only
        )
        
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Get embed_dim from config
        self.teacher_feature_dim = self.model.config.hidden_size
        
        self.normalize_transform = transforms.Normalize(mean=mean, std=std) # RT-DETRv4 does input downsampling.
        # self.avgpool_2x2 = nn.AvgPool2d(kernel_size=2, stride=2)

        _logger.info(f"[Teacher Model] DINOv3 loaded. Feature dimension: {self.teacher_feature_dim}")
        _logger.info(f"[Teacher Model] Model config: hidden_size={self.model.config.hidden_size}, "
                     f"num_attention_heads={self.model.config.num_attention_heads}, "
                     f"num_hidden_layers={self.model.config.num_hidden_layers}")
        
        # Verify weights loaded by checking param norm
        sample_param = next(self.model.parameters())
        param_norm = sample_param.norm().item()
        if param_norm < 1e-6:
            raise RuntimeError(f"[Teacher Model] Weight loading failed! Sample param norm is {param_norm:.6f} (near zero). "
                             f"Check if {dinov3_weights_path} contains valid weights.")
        _logger.info(f"[Teacher Model] Sample param norm: {param_norm:.4f} (valid)")

    def forward(self, images: torch.Tensor):
        """
        Extract patch token features from DINOv3 teacher model.
        
        Args:
            images: Input images [B, 3, H, W], already normalized by dataloader
            
        Returns:
            Feature map [B, C, H_patches, W_patches]
        """
        # Input: [B, 3, H, W]
        # We process at full resolution to get 1/16 features (P4)
        processed_images = images

        with torch.no_grad():
            # HuggingFace DINOv2/v3 output format
            outputs = self.model(processed_images, output_hidden_states=True)
            
            # Get the last hidden states (patch tokens + CLS + registers)
            # Shape: [B, num_tokens, hidden_size]
            last_hidden = outputs.last_hidden_state
            
            # Remove CLS token (first token) and any register tokens
            # DINOv2 with registers has: [CLS, patches..., registers...]
            # We only want the patch tokens
            num_patches_per_side = processed_images.shape[-1] // self.patch_size
            num_patches = num_patches_per_side * num_patches_per_side
            
            # Skip CLS token (index 0), take only patch tokens
            patch_tokens = last_hidden[:, 1:1+num_patches, :]  # [B, num_patches, hidden_size]
            
            B, N_patches, C_teacher = patch_tokens.shape
            H_patches = W_patches = int(N_patches ** 0.5)
            
            if H_patches * W_patches != N_patches:
                _logger.error(f"Number of patches {N_patches} is not a perfect square!")
                raise ValueError(f"Cannot reshape {N_patches} patches to HxW")
            
            # Reshape to feature map format [B, C, H, W]
            teacher_feature_map = patch_tokens.permute(0, 2, 1).reshape(B, C_teacher, H_patches, W_patches)

            return teacher_feature_map.detach()
