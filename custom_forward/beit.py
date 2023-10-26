import torch

from timm.models.beit import Beit
from .registry import register_method

_target_class = Beit


@register_method
def forward_features(self, x, requires_feat):
    feat = []
    x = self.patch_embed(x)
    x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    if self.pos_embed is not None:
        x = x + self.pos_embed
    x = self.pos_drop(x)

    rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
    for blk in self.blocks:
        x = blk(x, shared_rel_pos_bias=rel_pos_bias)
        feat.append(x)
    x = self.norm(x)
    return (x, feat) if requires_feat else x


@register_method
def forward(self, x, requires_feat=False):
    if requires_feat:
        x, feat = self.forward_features(x, requires_feat=True)
        x = self.forward_head(x, pre_logits=True)
        feat.append(x)
        x = self.head(x)
        return x, feat
    else:
        x = self.forward_features(x, requires_feat=False)
        x = self.forward_head(x)
        return x


@register_method
def stage_info(self, stage):
    if self.default_cfg['architecture'] == 'beitv2_base_patch16_224':
        shape = (197, 768)
        if stage == 1:
            index = 1
        elif stage == 2:
            index = 3
        elif stage == 3:
            index = 9
        elif stage == 4:
            index = 11
        elif stage == -1:
            index = -1
            shape = 768
        else:
            raise RuntimeError(f'Stage {stage} out of range (1-4)')
    elif self.default_cfg['architecture'] == 'beitv2_large_patch16_224':
        shape = (197, 1024)
        if stage == 1:
            index = 1
        elif stage == 2:
            index = 3
        elif stage == 3:
            index = 21
        elif stage == 4:
            index = 23
        elif stage == -1:
            index = -1
            shape = 1024
        else:
            raise RuntimeError(f'Stage {stage} out of range (1-4)')
    else:
        raise NotImplementedError(f'undefined stage_info() for model {self.default_cfg["architecture"]}')
    return index, shape
