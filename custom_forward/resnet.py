from timm.models.resnet import ResNet

from .registry import register_method

_target_class = ResNet


@register_method
def forward_features(self, x, requires_feat):
    feat = []
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.act1(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    feat.append(x)
    x = self.layer2(x)
    feat.append(x)
    x = self.layer3(x)
    feat.append(x)
    x = self.layer4(x)
    feat.append(x)

    return (x, feat) if requires_feat else x


@register_method
def forward(self, x, requires_feat=False):
    if requires_feat:
        x, feat = self.forward_features(x, requires_feat=True)
        x = self.forward_head(x, pre_logits=True)
        feat.append(x)
        x = self.fc(x)
        return x, feat
    else:
        x = self.forward_features(x, requires_feat=False)
        x = self.forward_head(x)
        return x


@register_method
def stage_info(self, stage):
    if self.default_cfg['architecture'] == 'resnet18':
        if stage == 1:
            index = 0
            shape = (64, 56, 56)
        elif stage == 2:
            index = 1
            shape = (128, 28, 28)
        elif stage == 3:
            index = 2
            shape = (256, 14, 14)
        elif stage == 4:
            index = 3
            shape = (512, 7, 7)
        elif stage == -1:
            index = -1
            shape = 512
        else:
            raise RuntimeError(f'Stage {stage} out of range (1-4)')
    elif self.default_cfg['architecture'] == 'resnet34':
        if stage == 1:
            index = 0
            shape = (64, 56, 56)
        elif stage == 2:
            index = 1
            shape = (128, 28, 28)
        elif stage == 3:
            index = 2
            shape = (256, 14, 14)
        elif stage == 4:
            index = 3
            shape = (512, 7, 7)
        elif stage == -1:
            index = -1
            shape = 512
        else:
            raise RuntimeError(f'Stage {stage} out of range (1-4)')
    elif self.default_cfg['architecture'] == 'resnet50':
        if stage == 1:
            index = 0
            shape = (256, 56, 56)
        elif stage == 2:
            index = 1
            shape = (512, 28, 28)
        elif stage == 3:
            index = 2
            shape = (1024, 14, 14)
        elif stage == 4:
            index = 3
            shape = (2048, 7, 7)
        elif stage == -1:
            index = -1
            shape = 2048
        else:
            raise RuntimeError(f'Stage {stage} out of range (1-4)')
    elif self.default_cfg['architecture'] == 'resnet101':
        if stage == 1:
            index = 0
            shape = (256, 56, 56)
        elif stage == 2:
            index = 1
            shape = (512, 28, 28)
        elif stage == 3:
            index = 2
            shape = (1024, 14, 14)
        elif stage == 4:
            index = 3
            shape = (2048, 7, 7)
        elif stage == -1:
            index = -1
            shape = 2048
        else:
            raise RuntimeError(f'Stage {stage} out of range (1-4)')
    else:
        raise NotImplementedError(f'undefined stage_info() for model {self.default_cfg["architecture"]}')
    return index, shape
