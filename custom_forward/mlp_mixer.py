from timm.models.mlp_mixer import MlpMixer
from .registry import register_method

_target_class = MlpMixer


@register_method
def forward(self, x, requires_feat=False):
    if requires_feat:
        x, feat = self.forward_features(x, requires_feat=True)
        if self.global_pool == 'avg':
            x = x.mean(dim=1)
        feat.append(x)
        x = self.head(x)
        return x, feat
    else:
        x = self.forward_features(x, requires_feat=False)
        if self.global_pool == 'avg':
            x = x.mean(dim=1)
        x = self.head(x)
        return x


@register_method
def forward_features(self, x, requires_feat):
    feat = []
    x = self.stem(x)
    if requires_feat:
        for blk in self.blocks:
            x = blk(x)
            feat.append(x)
    else:
        x = self.blocks(x)
    x = self.norm(x)
    return (x, feat) if requires_feat else x


@register_method
def stage_info(self, stage):
    if self.pretrained_cfg['architecture'] == 'mixer_b16_224':
        shape = (196, 768)
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
    elif self.pretrained_cfg['architecture'] == 'resmlp_12_224':
        shape = (196, 384)
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
            shape = 384
        else:
            raise RuntimeError(f'Stage {stage} out of range (1-4)')
    else:
        raise NotImplementedError(f'undefined stage_info() for model {self.pretrained_cfg["architecture"]}')
    return index, shape
