import torch

from ._base import BaseDistiller
from .registry import register_distiller


@register_distiller
class DIST(BaseDistiller):
    requires_feat = False

    def __init__(self, student, teacher, criterion, args, **kwargs):
        super(DIST, self).__init__(student, teacher, criterion, args)

    def forward(self, image, label, *args, **kwargs):
        with torch.no_grad():
            self.teacher.eval()
            logits_teacher = self.teacher(image)

        logits_student = self.student(image)

        loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, label)
        loss_kd = self.args.kd_loss_weight * dist_loss(logits_student, logits_teacher, self.args.dist_beta,
                                                       self.args.dist_gamma, self.args.dist_tau)
        losses_dict = {
            "loss_gt": loss_gt,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict


def dist_loss(logits_student, logits_teacher, beta=1., gamma=1., temperature=1.):
    y_s = (logits_student / temperature).softmax(dim=1)
    y_t = (logits_teacher / temperature).softmax(dim=1)
    inter_loss = temperature ** 2 * inter_class_relation(y_s, y_t)
    intra_loss = temperature ** 2 * intra_class_relation(y_s, y_t)
    return beta * inter_loss + gamma * intra_loss


def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))
