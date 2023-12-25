from timm.models.registry import register_model
from timm.models.swin_transformer import _create_swin_transformer

__all__ = ['swin_nano_patch4_window7_224', 'swin_pico_patch4_window7_224']


@register_model
def swin_nano_patch4_window7_224(pretrained=False, **kwargs):
    """ Swin-M @ 224x224, trained ImageNet-1k
    """
    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dim=64, depths=(2, 2, 2, 2), num_heads=(2, 4, 8, 16), **kwargs)
    return _create_swin_transformer('swin_nano_patch4_window7_224', pretrained=pretrained, **model_kwargs)


@register_model
def swin_pico_patch4_window7_224(pretrained=False, **kwargs):
    """ Swin-M @ 224x224, trained ImageNet-1k
    """
    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dim=48, depths=(2, 2, 2, 2), num_heads=(2, 4, 8, 16), **kwargs)
    return _create_swin_transformer('swin_pico_patch4_window7_224', pretrained=pretrained, **model_kwargs)
