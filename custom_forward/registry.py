import sys
from collections import defaultdict
from types import MethodType

_implemented_methods = defaultdict(dict)


def register_method(fn):
    mod = sys.modules[fn.__module__]
    target_class = getattr(mod, '_target_class', None)
    assert target_class is not None, f'please specify "_target_class" in "{mod}"!'
    target_method = fn.__name__
    _implemented_methods[target_class][target_method] = fn
    return fn


def register_new_forward(model):
    method_dict = _implemented_methods[model.__class__]
    for method in method_dict:
        fn = method_dict[method]
        setattr(model, method, MethodType(fn, model))
