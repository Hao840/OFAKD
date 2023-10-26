import torch.nn.functional as F

from custom_model import MobileNetV1
from .registry import register_method

_target_class = MobileNetV1


@register_method
def forward(self, x, requires_feat=False):
    feat = []
    x = self.model[3][:-1](self.model[0:3](x))
    feat.append(x)
    x = self.model[5][:-1](self.model[4:5](F.relu(x)))
    feat.append(x)
    x = self.model[11][:-1](self.model[6:11](F.relu(x)))
    feat.append(x)
    x = self.model[13][:-1](self.model[12:13](F.relu(x)))
    feat.append(x)
    x = self.model[14](F.relu(x))
    x = x.reshape(-1, 1024)
    feat.append(x)
    x = self.fc(x)
    return (x, feat) if requires_feat else x


@register_method
def stage_info(self, stage):
    if self.default_cfg['architecture'] == 'mobilenetv1':
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
        raise NotImplementedError(f'undefined stage_info() for model {self.default_cfg["architecture"]}')
    return index, shape
