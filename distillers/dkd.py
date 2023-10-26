import torch
import torch.nn.functional as F

from ._base import BaseDistiller
from .registry import register_distiller


@register_distiller
class DKD(BaseDistiller):
    requires_feat = False

    def __init__(self, student, teacher, criterion, args, **kwargs):
        super(DKD, self).__init__(student, teacher, criterion, args)

    def forward(self, image, label, *args, **kwargs):
        with torch.no_grad():
            self.teacher.eval()
            logits_teacher = self.teacher(image)

        logits_student = self.student(image)

        if len(label.shape) == 2:  # mixup / smoothing
            target = label.max(1)[1]
        else:
            target = label

        loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, label)
        loss_kd = self.args.kd_loss_weight * dkd_loss(logits_student, logits_teacher, target,
                                                      self.dkd_alpha, self.dkd_beta, self.dkd_temperature)
        losses_dict = {
            "loss_gt": loss_gt,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict


def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
            F.kl_div(log_pred_student, pred_teacher, reduction='batchmean')
            * (temperature ** 2)
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
            F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='batchmean')
            * (temperature ** 2)
    )
    return alpha * tckd_loss + beta * nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt
