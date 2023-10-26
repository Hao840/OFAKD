from timm.models.efficientnet import EfficientNet
from .registry import register_method

_target_class = EfficientNet


@register_method
def forward(self, x, requires_feat=False):
    if requires_feat:
        x, feat = self.forward_features(x, requires_feat=True)
        x = self.forward_head(x, pre_logits=True)
        feat.append(x)
        x = self.classifier(x)
        return x, feat
    else:
        x = self.forward_features(x, requires_feat=False)
        x = self.forward_head(x)
        return x


@register_method
def forward_features(self, x, requires_feat):
    feat = []
    x = self.conv_stem(x)
    x = self.bn1(x)
    if requires_feat:
        for blks in self.blocks:
            for blk in blks:
                x = blk(x)
                feat.append(x)
    else:
        x = self.blocks(x)
    x = self.conv_head(x)
    x = self.bn2(x)
    feat.append(x)  # idx=17, for debug
    return (x, feat) if requires_feat else x


@register_method
def stage_info(self, stage):
    if self.default_cfg['architecture'] == 'mobilenetv2_100':
        if stage == 1:
            index = 2
            shape = (24, 56, 56)
        elif stage == 2:
            index = 5
            shape = (32, 28, 28)
        elif stage == 3:
            index = 12
            shape = (96, 14, 14)
        elif stage == 4:
            # index = 16
            index = 17
            shape = (1280, 7, 7)
        elif stage == -1:
            index = -1
            shape = 1280
        else:
            raise RuntimeError(f'Stage {stage} out of range (1-4)')
    else:
        raise NotImplementedError(f'undefined stage_info() for model {self.default_cfg["architecture"]}')
    return index, shape
