import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import Block
from ._base import BaseDistiller
from .registry import register_distiller
from .utils import GAP1d, get_module_dict, init_weights, is_cnn_model, PatchMerging, SepConv, set_module_dict, \
    TokenFilter, TokenFnContext


def ofa_loss(logits_student, logits_teacher, target_mask, eps, temperature=1.):
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    prod = (pred_teacher + target_mask) ** eps
    loss = torch.sum(- (prod - target_mask) * torch.log(pred_student), dim=-1)
    return loss.mean()


@register_distiller
class OFA(BaseDistiller):
    requires_feat = True

    def __init__(self, student, teacher, criterion, args, **kwargs):
        super(OFA, self).__init__(student, teacher, criterion, args)

        if len(self.args.ofa_eps) == 1:
            eps = [self.args.ofa_eps[0] for _ in range(len(self.args.ofa_stage) + 1)]
            self.args.ofa_eps = eps

        assert len(self.args.ofa_stage) + 1 == len(self.args.ofa_eps)  # +1 for logits

        self.projector = nn.ModuleDict()

        is_cnn_student = is_cnn_model(student)

        _, feature_dim_t = self.teacher.stage_info(-1)
        _, feature_dim_s = self.student.stage_info(-1)

        for stage in self.args.ofa_stage:
            _, size_s = self.student.stage_info(stage)

            if is_cnn_student:
                in_chans, _, _ = size_s

                if stage != 4:
                    down_sample_blk_num = 4 - stage
                    down_sample_blks = []
                    for i in range(down_sample_blk_num):
                        if i == down_sample_blk_num - 1:
                            out_chans = max(feature_dim_s, feature_dim_t)
                        else:
                            out_chans = in_chans * 2
                        down_sample_blks.append(SepConv(in_chans, out_chans))
                        in_chans *= 2
                else:
                    down_sample_blks = [nn.Conv2d(in_chans, max(feature_dim_s, feature_dim_t), 1, 1, 0)]

                projector = nn.Sequential(
                    *down_sample_blks,
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(max(feature_dim_s, feature_dim_t), args.num_classes)  # todo: cifar100
                )
            else:
                patch_num, embed_dim = size_s
                token_num = getattr(student, 'num_tokens', 0)  # cls tokens

                final_patch_grid = 7  # finally there are 49 patches
                patch_grid = int(patch_num ** .5)
                merge_num = max(int(np.log2(patch_grid / final_patch_grid)), 0)
                merger_modules = []
                for i in range(merge_num):
                    if i == 0:  # proj to feature_dim_s
                        merger_modules.append(
                            PatchMerging(input_resolution=(patch_grid // 2 ** i, patch_grid // 2 ** i),
                                         dim=embed_dim,
                                         out_dim=feature_dim_s,
                                         act_layer=nn.GELU))
                    else:
                        merger_modules.append(
                            PatchMerging(input_resolution=(patch_grid // 2 ** i, patch_grid // 2 ** i),
                                         dim=feature_dim_s,
                                         out_dim=feature_dim_s,
                                         act_layer=nn.GELU if i != merge_num - 1 else nn.Identity))
                patch_merger = nn.Sequential(*merger_modules)
                blocks = nn.Sequential(
                    *[Block(dim=feature_dim_s, num_heads=4) for _ in range(max(4 - stage, 1))]  # todo: check this
                )
                if token_num != 0:
                    get_feature = nn.Sequential(
                        TokenFilter(token_num, remove_mode=False),  # todo: token_num > 1
                        nn.Flatten()
                    )
                else:
                    get_feature = GAP1d()
                projector = nn.Sequential(
                    TokenFnContext(token_num, patch_merger),
                    blocks,
                    get_feature,
                    nn.Linear(feature_dim_s, args.num_classes)  # todo: cifar100
                )
            set_module_dict(self.projector, stage, projector)
        self.projector.apply(init_weights)
        # print(self.projector)  # for debug

    def forward(self, image, label, *args, **kwargs):
        with torch.no_grad():
            self.teacher.eval()
            logits_teacher = self.teacher(image)

        logits_student, feat_student = self.student(image, requires_feat=True)

        num_classes = logits_student.size(-1)
        if len(label.shape) != 1:  # label smoothing
            target_mask = F.one_hot(label.argmax(-1), num_classes)
        else:
            target_mask = F.one_hot(label, num_classes)

        ofa_losses = []
        for stage, eps in zip(self.args.ofa_stage, self.args.ofa_eps):
            idx_s, _ = self.student.stage_info(stage)
            feat_s = feat_student[idx_s]
            logits_student_head = get_module_dict(self.projector, stage)(feat_s)

            ofa_losses.append(
                ofa_loss(logits_student_head, logits_teacher, target_mask, eps, self.args.ofa_temperature))

        loss_ofa = self.args.ofa_loss_weight * sum(ofa_losses)

        loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, label)
        loss_kd = self.args.kd_loss_weight * ofa_loss(logits_student, logits_teacher, target_mask,
                                                      self.args.ofa_eps[-1], self.args.ofa_temperature)
        losses_dict = {
            "loss_gt": loss_gt,
            "loss_kd": loss_kd,
            "loss_ofa": loss_ofa
        }
        return logits_student, losses_dict
