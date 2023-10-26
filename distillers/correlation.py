import torch
import torch.nn as nn

from ._base import BaseDistiller
from .registry import register_distiller


@register_distiller
class Correlation(BaseDistiller):
    requires_feat = True

    def __init__(self, student, teacher, criterion, args, **kwargs):
        super(Correlation, self).__init__(student, teacher, criterion, args)
        feat_s_channel = student.stage_info(-1)[1]
        feat_t_channel = teacher.stage_info(-1)[1]
        self.embed_s = LinearEmbed(feat_s_channel, self.args.correlation_feat_dim)
        self.embed_t = LinearEmbed(feat_t_channel, self.args.correlation_feat_dim)

    def forward(self, image, label, *args, **kwargs):
        with torch.no_grad():
            self.teacher.eval()
            logits_teacher, feat_teacher = self.teacher(image, requires_feat=True)

        logits_student, feat_student = self.student(image, requires_feat=True)

        f_s = self.embed_s(feat_student[-1])
        f_t = self.embed_t(feat_teacher[-1])
        delta = torch.abs(f_s - f_t)

        loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, label)
        loss_kd = self.args.kd_loss_weight * self.args.correlation_scale * torch.mean((delta[:-1] * delta[1:]).sum(1))
        losses_dict = {
            "loss_gt": loss_gt,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict


class LinearEmbed(nn.Module):
    def __init__(self, dim_in=1024, dim_out=128):
        super(LinearEmbed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x
