import torch.nn as nn

from .registry import register_distiller


class BaseDistiller(nn.Module):
    def __init__(self, student, teacher, criterion, args):
        super(BaseDistiller, self).__init__()
        self.student = student
        self.teacher = teacher
        self.criterion = criterion
        self.args = args

    def forward(self, image, label, *args, **kwargs):
        raise NotImplementedError

    def get_learnable_parameters(self):
        student_params = 0
        extra_params = 0
        for n, p in self.named_parameters():
            if n.startswith('student'):
                student_params += p.numel()
            elif n.startswith('teacher'):
                continue
            else:
                if p.requires_grad:
                    extra_params += p.numel()
        return student_params, extra_params



@register_distiller
class Vanilla(BaseDistiller):
    requires_feat = False

    def __init__(self, student, teacher, criterion, args, **kwargs):
        super(Vanilla, self).__init__(student, teacher, criterion, args)

    def forward(self, image, label, *args, **kwargs):
        logits_student = self.student(image)

        loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, label)
        losses_dict = {
            "loss_gt": loss_gt,
        }
        return logits_student, losses_dict
