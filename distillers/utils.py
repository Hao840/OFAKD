import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import _assert, trunc_normal_


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, out_dim=None, norm_layer=nn.LayerNorm, act_layer=nn.Identity):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim or 2 * dim
        self.norm = norm_layer(4 * dim)

        in_features = 4 * dim
        self.reduction = nn.Linear(in_features, self.out_dim, bias=False)
        self.act = act_layer()

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        _assert(L == H * W, "input feature has wrong size")
        _assert(H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even.")

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)
        x = self.act(x)

        return x


class GAP1d(nn.Module):
    def __init__(self):
        super(GAP1d, self).__init__()

    def forward(self, x):
        return x.mean(1)


class TokenFilter(nn.Module):
    """remove cls tokens in forward"""

    def __init__(self, number=1, inverse=False, remove_mode=True):
        super(TokenFilter, self).__init__()
        self.number = number
        self.inverse = inverse
        self.remove_mode = remove_mode

    def forward(self, x):
        if self.inverse and self.remove_mode:
            x = x[:, :-self.number, :]
        elif self.inverse and not self.remove_mode:
            x = x[:, -self.number:, :]
        elif not self.inverse and self.remove_mode:
            x = x[:, self.number:, :]
        else:
            x = x[:, :self.number, :]
        return x


class TokenFnContext(nn.Module):
    def __init__(self, token_num=0, fn: nn.Module = nn.Identity(), token_fn: nn.Module = nn.Identity(), inverse=False):
        super(TokenFnContext, self).__init__()
        self.token_num = token_num
        self.fn = fn
        self.token_fn = token_fn
        self.inverse = inverse
        self.token_filter = TokenFilter(number=token_num, inverse=inverse, remove_mode=False)
        self.feature_filter = TokenFilter(number=token_num, inverse=inverse)

    def forward(self, x):
        tokens = self.token_filter(x)
        features = self.feature_filter(x)
        features = self.fn(features)
        if self.token_num == 0:
            return features

        tokens = self.token_fn(tokens)
        if self.inverse:
            x = torch.cat([features, tokens], dim=1)
        else:
            x = torch.cat([tokens, features], dim=1)
        return x


class LambdaModule(nn.Module):
    def __init__(self, lambda_fn):
        super(LambdaModule, self).__init__()
        self.fn = lambda_fn

    def forward(self, x):
        return self.fn(x)


class MyPatchMerging(nn.Module):
    def __init__(self, out_patch_num):
        super().__init__()
        self.out_patch_num = out_patch_num

    def forward(self, x):
        B, L, D = x.shape
        patch_size = int(L ** 0.5)
        assert patch_size ** 2 == L
        out_patch_size = int(self.out_patch_num ** 0.5)
        assert out_patch_size ** 2 == self.out_patch_num
        grid_size = patch_size // out_patch_size
        assert grid_size * out_patch_size == patch_size
        x = x.view(B, out_patch_size, grid_size, out_patch_size, grid_size, D)
        x = torch.einsum('bhpwqd->bhwpqd', x)
        x = x.reshape(shape=(B, out_patch_size ** 2, -1))
        return x


class SepConv(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        #   depthwise and pointwise convolution, downsample by 2
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in,
                      bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)


def init_weights(module):
    for n, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def kd_loss(logits_student, logits_teacher, temperature=1.):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction='batchmean')
    loss_kd *= temperature ** 2
    return loss_kd


def is_cnn_model(distiller):
    if hasattr(distiller, 'module'):
        _, sizes = distiller.module.stage_info(1)
    else:
        _, sizes = distiller.stage_info(1)
    if len(sizes) == 3:  # C H W
        return True
    elif len(sizes) == 2:  # L D
        return False
    else:
        raise RuntimeError('unknown model feature shape')


def set_module_dict(module_dict, k, v):
    if not isinstance(k, str):
        k = str(k)
    module_dict[k] = v


def get_module_dict(module_dict, k):
    if not isinstance(k, str):
        k = str(k)
    return module_dict[k]


def patchify(imgs, p):
    """
    imgs: (N, C, H, W)
    x: (N, L, patch_size**2, C)
    """
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    in_chans = imgs.shape[1]
    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], in_chans, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * in_chans))
    return x


class Unpatchify(nn.Module):
    def __init__(self, p):
        super(Unpatchify, self).__init__()
        self.p = p

    def forward(self, x):
        return _unpatchify(x, self.p)


def _unpatchify(x, p):
    """
    x: (N, L, patch_size**2 *C)
    imgs: (N, C, H, W)
    """
    # p = self.patch_embed.patch_size[0]
    h = w = int(x.shape[1] ** .5)
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, p, p, -1))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], -1, h * p, h * p))
    return imgs




