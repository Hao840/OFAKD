from timm.models.convnext import ConvNeXt
from .registry import register_method

_target_class = ConvNeXt


@register_method
def forward(self, x, requires_feat=False):
    if requires_feat:
        x, feat = self.forward_features(x, requires_feat=True)
        x = self.forward_head(x, pre_logits=True)
        feat.append(x)
        x = self.head.fc(x)
        return x, feat
    else:
        x = self.forward_features(x, requires_feat=False)
        x = self.forward_head(x)
        return x


@register_method
def forward_features(self, x, requires_feat):
    feat = []
    x = self.stem(x)
    if requires_feat:
        for stage in self.stages:
            x = stage(x)
            feat.append(x)
    else:
        x = self.stages(x)
    x = self.norm_pre(x)
    return (x, feat) if requires_feat else x


@register_method
def stage_info(self, stage):
    if self.pretrained_cfg['architecture'] == 'convnext_tiny':
        if stage == 1:
            index = 0
            shape = (96, 56, 56)
        elif stage == 2:
            index = 1
            shape = (192, 28, 28)
        elif stage == 3:
            index = 2
            shape = (384, 14, 14)
        elif stage == 4:
            index = 3
            shape = (768, 7, 7)
        elif stage == -1:
            index = -1
            shape = 768
        else:
            raise RuntimeError(f'Stage {stage} out of range (1-4)')
    elif self.pretrained_cfg['architecture'] == 'convnext_base':
        if stage == 1:
            index = 0
            shape = (128, 56, 56)
        elif stage == 2:
            index = 1
            shape = (256, 28, 28)
        elif stage == 3:
            index = 2
            shape = (512, 14, 14)
        elif stage == 4:
            index = 3
            shape = (1024, 7, 7)
        elif stage == -1:
            index = -1
            shape = 1024
        else:
            raise RuntimeError(f'Stage {stage} out of range (1-4)')
    else:
        raise NotImplementedError(f'undefined stage_info() for model {self.pretrained_cfg["architecture"]}')
    return index, shape
