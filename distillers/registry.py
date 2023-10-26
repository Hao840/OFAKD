from collections import OrderedDict


_distiller_dict = OrderedDict()


def register_distiller(fn):
    module_name = fn.__name__.lower()
    _distiller_dict[module_name] = fn
    return fn


def get_distiller(name):
    return _distiller_dict[name.lower()]