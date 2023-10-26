from custom_model import CycleNet
from .registry import register_method

_target_class = CycleNet


@register_method
def forward(self, x, requires_feat=False):
    feats = []
    x = self.forward_embeddings(x)

    for idx, block in enumerate(self.network):
        x = block(x)
        feats.append(x.permute(0, 3, 1, 2).contiguous())

    B, H, W, C = x.shape
    x = x.reshape(B, -1, C)

    x = self.norm(x)
    x = x.mean(1)
    feats.append(x)

    if requires_feat:
        return self.head(x), feats
    else:
        return self.head(x)


@register_method
def stage_info(self, stage):
    if self.default_cfg['architecture'] == 'CycleMLP_B3':
        if stage == 1:
            index = 0
            shape = (64, 56, 56)
        elif stage == 2:
            index = 2
            shape = (128, 28, 28)
        elif stage == 3:
            index = 4
            shape = (320, 14, 14)
        elif stage == 4:
            index = 6
            shape = (512, 7, 7)
        elif stage == -1:
            index = -1
            shape = 512
        else:
            raise RuntimeError(f'Stage {stage} out of range (1-4)')
    else:
        raise NotImplementedError(f'undefined stage_info() for model {self.default_cfg["architecture"]}')
    return index, shape
