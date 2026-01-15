"""
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import torch.nn as nn
from ..core import register


__all__ = ['DEIM', ]


@register()
class DEIM(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder

    def forward(self, x, targets=None, teacher_encoder_output=None):
        x = self.backbone(x)
        
        encoder_output = self.encoder(x)
        
        student_distill_output = None
        student_distill_output = None
        if isinstance(encoder_output, tuple) and len(encoder_output) == 2:
            x_fpn_features, student_distill_output = encoder_output
        else:
            x_fpn_features = encoder_output

        x_decoder_out = self.decoder(x_fpn_features, targets)

        if student_distill_output is not None:
             x_decoder_out['student_distill_output'] = student_distill_output
        
        if teacher_encoder_output is not None:
             x_decoder_out['teacher_encoder_output'] = teacher_encoder_output

        return x_decoder_out

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
