"""
DSViT: Double-Stage Vision Transformer for Object Detection
Combines MBConv local processing (Stage 1) with ViT global reasoning (Stage 2).
Strictly follows vit_tiny.py implementation details for the Transformer part.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, LayerScale, trunc_normal_


import numpy as np
from functools import partial
from typing import Literal, Tuple
from ..core import register

# --------------------------------------------------------------------------------
# ViT Components from vit_tiny.py
# --------------------------------------------------------------------------------

class RopePositionEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        head_dim = embed_dim // num_heads
        assert head_dim % 4 == 0, "Head dimension must be divisible by 4 for 2D RoPE"
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")

        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = head_dim
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords
        self.dtype = dtype
        self.register_buffer(
            "periods",
            torch.empty(head_dim // 4, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def forward(self, *, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.periods.device
        dtype = self.dtype if self.dtype is not None else torch.get_default_dtype()
        dd = {"device": device, "dtype": dtype}

        if self.normalize_coords == "max":
            max_HW = max(H, W)
            coords_h = torch.arange(0.5, H, **dd) / max_HW
            coords_w = torch.arange(0.5, W, **dd) / max_HW
        elif self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, H, **dd) / H
            coords_w = torch.arange(0.5, W, **dd) / W
        else: # min
            min_HW = min(H, W)
            coords_h = torch.arange(0.5, H, **dd) / min_HW
            coords_w = torch.arange(0.5, W, **dd) / min_HW

        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
        coords = coords.flatten(0, 1)
        coords = 2.0 * coords - 1.0

        if self.training and self.shift_coords is not None:
            coords += torch.empty(2, **dd).uniform_(-self.shift_coords, self.shift_coords)[None, :]
        if self.training and self.jitter_coords is not None:
            jitter = (torch.empty(2, **dd).uniform_(-np.log(self.jitter_coords), np.log(self.jitter_coords))).exp()
            coords *= jitter[None, :]
        if self.training and self.rescale_coords is not None:
            rescale = (torch.empty(1, **dd).uniform_(-np.log(self.rescale_coords), np.log(self.rescale_coords))).exp()
            coords *= rescale

        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2).repeat(1, 2)

        sin = torch.sin(angles)
        cos = torch.cos(angles)
        return sin.unsqueeze(0).unsqueeze(0), cos.unsqueeze(0).unsqueeze(0)

    def _init_weights(self):
        device = self.periods.device
        dtype = self.dtype if self.dtype is not None else torch.get_default_dtype()
        if self.base is not None:
            periods = self.base ** (2 * torch.arange(self.D_head // 4, device=device, dtype=dtype) / (self.D_head // 2))
        else:
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, self.D_head // 4, device=device, dtype=dtype)
            periods = self.max_period * (base ** (exponents - 1))
        self.periods.data.copy_(periods)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x, sin, cos):
    return (x * cos) + (rotate_half(x) * sin)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope_sincos=None):
        B, N, C = x.shape
        # qkv: [B, N, 3*C] -> [B, N, 3, H, D] -> [3, B, H, N, D]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if rope_sincos is not None:
            sin, cos = rope_sincos
            # Split CLS and Patch tokens
            # Assumes x has cls_token at index 0!
            # If no cls token, direct apply. But DSViT usually will have one.
            if x.shape[1] > sin.shape[2]: # Has CLS token (N > H*W)
                 q_cls, q_patch = q[:, :, :1, :], q[:, :, 1:, :]
                 k_cls, k_patch = k[:, :, :1, :], k[:, :, 1:, :]
                 q_patch = apply_rope(q_patch, sin, cos)
                 k_patch = apply_rope(k_patch, sin, cos)
                 q = torch.cat((q_cls, q_patch), dim=2)
                 k = torch.cat((k_cls, k_patch), dim=2)
            else:
                 q = apply_rope(q, sin, cos)
                 k = apply_rope(k, sin, cos)

        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop)
        x = x.transpose(1, 2).reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x); x = self.fc2(x); x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, rope_sincos=None):
        attn_output = self.attn(self.norm1(x), rope_sincos=rope_sincos)
        x = x + self.drop_path(attn_output)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# --------------------------------------------------------------------------------
# DSViT Components
# --------------------------------------------------------------------------------

class HierarchicalStem(nn.Module):
    def __init__(self, in_ch=3, out_ch=64, act_layer=nn.GELU):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch // 2, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch // 2)
        self.act1 = act_layer()
        self.conv2 = nn.Conv2d(out_ch // 2, out_ch, 3, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = act_layer()
        self.conv3 = nn.Conv2d(out_ch, out_ch, 3, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.act3 = act_layer()

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        return x

class MBConvBlock(nn.Module):
    def __init__(self, in_chs, out_chs, correlation, stride=1, act_layer=nn.GELU, drop_path=0.):
        super().__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.stride = stride
        mid_chs = int(in_chs * correlation)

        self.conv1 = nn.Conv2d(in_chs, mid_chs, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_chs)
        self.act1 = act_layer()

        self.conv2 = nn.Conv2d(mid_chs, mid_chs, 3, stride, 1, groups=mid_chs, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_chs)
        self.act2 = act_layer()

        self.conv3 = nn.Conv2d(mid_chs, out_chs, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_chs)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.shortcut = stride == 1 and in_chs == out_chs

    def forward(self, x):
        shortcut = x
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if self.shortcut:
            x = shortcut + self.drop_path(x)
        return x

# --------------------------------------------------------------------------------
# DSViT Main Class
# --------------------------------------------------------------------------------

@register()
class DSViT(nn.Module):
    def __init__(self, 
                 in_ch=3, 
                 embed_dims=[128, 192], 
                 depths=[2, 10],
                 num_heads=6,
                 mlp_ratio=3.,
                 drop_path_rate=0.1,
                 act_layer='gelu',
                 out_indices=[0, 1],
                 pretrained=None  # Add pretrained arg
                 ):
        super().__init__()
        
        act_layer_func = nn.GELU if act_layer == 'gelu' else nn.ReLU
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # 1. Stem (Down to 1/8)
        self.stem = HierarchicalStem(in_ch, embed_dims[0], act_layer=act_layer_func)

        # Shared progressive drop path
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # 2. Stage 1: Local MBConv (at 1/8)
        self.stage1 = nn.Sequential(*[
            MBConvBlock(embed_dims[0], embed_dims[0], correlation=3, stride=1, 
                        act_layer=act_layer_func, drop_path=dpr[i])
            for i in range(depths[0])
        ])

        # 3. Transition: Downsample to 1/16
        self.transition = nn.Sequential(
            nn.Conv2d(embed_dims[0], embed_dims[1], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims[1]) 
        )

        # 4. Stage 2: Global ViT (at 1/16)
        vit_dpr = dpr[depths[0]:]
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[1]))
        trunc_normal_(self.cls_token, std=.02)

        self.stage2 = nn.ModuleList([
            Block(
                dim=embed_dims[1], num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, 
                drop=0., attn_drop=0., drop_path=p, norm_layer=norm_layer, act_layer=act_layer_func
            )
            for p in vit_dpr
        ])

        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dims[1], num_heads=num_heads, base=100.0,
            normalize_coords="separate", shift_coords=None, jitter_coords=None,
            rescale_coords=None, dtype=None, device=None,
        )

        # P3 fusion: concat MBConv(128) + ViT↑2x(192) → 192 channels
        # Add learnable fusion for 1/8 resolution where local details matter most
        self.embed_dims = embed_dims 
        self.apply(self._init_weights)

        if pretrained:
            self.load_pretrained_weights(pretrained)

    def load_pretrained_weights(self, pretrained_path):
        import logging
        logger = logging.getLogger(__name__)
        
        if not pretrained_path:
            return

        try:
            logger.info(f"Loading pretrained weights from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            state_dict = checkpoint
            
            # If state_dict is inside a key (e.g. 'model' or 'state_dict')
            if 'model' in state_dict:
                state_dict = state_dict['model']
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            new_dict = {}
            loaded_keys = []
            
            # Mapping rules:
            # ViT Tiny checkpoint: blocks.X -> DSViT: stage2.X
            # ViT Tiny checkpoint: rope_embed -> DSViT: rope_embed
            
            for k, v in state_dict.items():
                if 'blocks.' in k:
                    # Handle both 'blocks.' and '_model.blocks.' cases
                    if '_model.blocks.' in k:
                        new_key = k.replace('_model.blocks.', 'stage2.')
                    elif k.startswith('blocks.'):
                        new_key = k.replace('blocks.', 'stage2.')
                    else:
                        continue # Should not happen if 'blocks.' in k
                    
                    # We only load if shapes match. 
                    new_dict[new_key] = v
                    loaded_keys.append(new_key)
                    
                elif 'rope_embed.' in k:
                     # Handle both 'rope_embed.' and '_model.rope_embed.'
                    if '_model.rope_embed.' in k:
                        new_key = k.replace('_model.rope_embed.', 'rope_embed.')
                    else:
                        new_key = k  # Assume it is already 'rope_embed.xxx'
                        
                    new_dict[new_key] = v
                    loaded_keys.append(new_key)
                    
            # Load into model
            if len(new_dict) == 0:
                logger.warning(f"No keys matched! Checkpoint keys sample: {list(state_dict.keys())[:5]}")
            else:
                msg = self.load_state_dict(new_dict, strict=False)
                logger.info(f"Pretrained weights loaded. Mapped {len(new_dict)} keys. Missing keys: {msg.missing_keys}")
            
        except Exception as e:
            logger.warning(f"Failed to load pretrained weights: {e}")

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
                
    def forward(self, x):
        # 1. Stem -> 1/8
        x = self.stem(x)
        
        # 2. Stage 1 (MBConv) -> 1/8
        feat_s1 = self.stage1(x)
        
        # 3. Transition -> 1/16
        x = self.transition(feat_s1)
        
        # 4. Stage 2 (ViT) -> 1/16
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2) # [B, N, C]
        
        # Add CLS token
        cls_token = self.cls_token.expand(B, -1, -1)
        x_flat = torch.cat((cls_token, x_flat), dim=1)
        
        # RoPE
        rope_sincos = self.rope_embed(H=H, W=W)
        
        for blk in self.stage2:
            x_flat = blk(x_flat, rope_sincos=rope_sincos)
            
        # Remove CLS token and reshape
        feat_s2 = x_flat[:, 1:].transpose(1, 2).reshape(B, C, H, W).contiguous()
        
        # 5. Feature Pyramid
        p4 = feat_s2
        p5 = F.interpolate(feat_s2, scale_factor=0.5, mode='bilinear', align_corners=False)
        
        # P3: Direct output from Stage 1 (MBConv features, 128 ch)
        # Detailed spatial features, will be fused with P4 in HybridEncoder
        p3 = feat_s1
        
        return p3, p4, p5
