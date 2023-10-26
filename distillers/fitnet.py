import torch
import torch.nn.functional as F
from torch import nn

from ._base import BaseDistiller
from .registry import register_distiller
from .utils import get_module_dict, init_weights, is_cnn_model, set_module_dict


@register_distiller
class FitNet(BaseDistiller):
    requires_feat = True

    def __init__(self, student, teacher, criterion, args, **kwargs):
        super(FitNet, self).__init__(student, teacher, criterion, args)

        assert is_cnn_model(student) and is_cnn_model(teacher), 'current FitNet implementation only support cnn models!'

        self.projector = nn.ModuleDict()

        for stage in self.args.fitnet_stage:
            _, size_s = self.student.stage_info(stage)
            _, size_t = self.teacher.stage_info(stage)

            in_chans_s, _, _ = size_s
            in_chans_t, _, _ = size_t

            projector = nn.Conv2d(in_chans_s, in_chans_t, 1, 1, 0, bias=False)
            set_module_dict(self.projector, stage, projector)

        self.projector.apply(init_weights)

    def forward(self, image, label, *args, **kwargs):
        with torch.no_grad():
            self.teacher.eval()
            logits_teacher, feat_teacher = self.teacher(image, requires_feat=True)

        logits_student, feat_student = self.student(image, requires_feat=True)

        fitnet_losses = []
        for stage in self.args.fitnet_stage:
            idx_s, _ = self.student.stage_info(stage)
            idx_t, _ = self.teacher.stage_info(stage)

            feat_s = get_module_dict(self.projector, stage)(feat_student[idx_s])
            feat_t = feat_teacher[idx_t]

            fitnet_losses.append(F.mse_loss(feat_s, feat_t))

        loss_fitnet = self.args.fitnet_loss_weight * sum(fitnet_losses)
        loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, label)

        losses_dict = {
            "loss_gt": loss_gt,
            "loss_fitnet": loss_fitnet
        }
        return logits_student, losses_dict
