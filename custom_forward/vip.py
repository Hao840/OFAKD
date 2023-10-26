from custom_model import VisionPermutator
from .registry import register_method

_target_class = VisionPermutator


@register_method
def forward(self, x, requires_feat=False):
    feats = []
    x = self.forward_embeddings(x)

    for idx, block in enumerate(self.network):
        x = block(x)
        feats.append(x.view(x.size(0), -1, x.size(-1)))

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
    if self.default_cfg['architecture'] == 'vip_s7':
        if stage == 1:
            index = 0
            shape = (1024, 192)
        elif stage == 2:
            index = 2
            shape = (256, 384)
        elif stage == 3:
            index = 3
            shape = (256, 384)
        elif stage == 4:
            index = 4
            shape = (256, 384)
        elif stage == -1:
            index = -1
            shape = 384
        else:
            raise RuntimeError(f'Stage {stage} out of range (1-4)')
    else:
        raise NotImplementedError(f'undefined stage_info() for model {self.default_cfg["architecture"]}')
    return index, shape
