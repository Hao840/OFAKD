from timm.models.swin_transformer import SwinTransformer
from .registry import register_method

_target_class = SwinTransformer


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
def forward_features(self, x, requires_feat):
    feat = []
    x = self.patch_embed(x)
    if self.absolute_pos_embed is not None:
        x = x + self.absolute_pos_embed
    x = self.pos_drop(x)
    for layers in self.layers:
        for layer in layers.blocks:
            x = layer(x)
            feat.append(x)
        if layers.downsample is not None:
            x = layers.downsample(x)
    x = self.norm(x)  # B L C
    feat.append(x)  # idx=12/24, for debug
    return (x, feat) if requires_feat else x


@register_method
def stage_info(self, stage):
    if self.default_cfg['architecture'] == 'swin_tiny_patch4_window7_224':
        if stage == 1:
            index = 1
            shape = (3136, 96)
        elif stage == 2:
            index = 3
            shape = (784, 192)
        elif stage == 3:
            index = 9
            shape = (196, 384)
        elif stage == 4:
            index = 11
            shape = (49, 768)
        elif stage == -1:
            index = -1
            shape = 768
        else:
            raise RuntimeError(f'Stage {stage} out of range (1-4)')
    elif self.default_cfg['architecture'] == 'swin_base_patch4_window7_224':
        if stage == 1:
            index = 1
            shape = (3136, 128)
        elif stage == 2:
            index = 3
            shape = (784, 256)
        elif stage == 3:
            index = 21
            shape = (196, 512)
        elif stage == 4:
            index = 23
            shape = (49, 1024)
        elif stage == -1:
            index = -1
            shape = 1024
        else:
            raise RuntimeError(f'Stage {stage} out of range (1-4)')
    elif self.default_cfg['architecture'] == 'swin_nano_patch4_window7_224':
        if stage == 1:
            index = 1
            shape = (3136, 64)
        elif stage == 2:
            index = 3
            shape = (784, 128)
        elif stage == 3:
            index = 5
            shape = (196, 256)
        elif stage == 4:
            index = 7
            shape = (49, 512)
        elif stage == -1:
            index = -1
            shape = 512
        else:
            raise RuntimeError(f'Stage {stage} out of range (1-4)')
    elif self.default_cfg['architecture'] == 'swin_pico_patch4_window7_224':
        if stage == 1:
            index = 1
            shape = (3136, 48)
        elif stage == 2:
            index = 3
            shape = (784, 96)
        elif stage == 3:
            index = 5
            shape = (196, 192)
        elif stage == 4:
            index = 7
            shape = (49, 384)
        elif stage == -1:
            index = -1
            shape = 384
        else:
            raise RuntimeError(f'Stage {stage} out of range (1-4)')
    else:
        raise NotImplementedError(f'undefined stage_info() for model {self.default_cfg["architecture"]}')
    return index, shape
