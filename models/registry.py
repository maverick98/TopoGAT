#models/registry.py
MODEL_REGISTRY = {}
TOPOGAT_REGISTRY = {}
TOPOGIN_REGISTRY = {}

def register_topogat(name):
    def wrapper(cls):
        TOPOGAT_REGISTRY[name.lower()] = cls
        return cls
    return wrapper

def register_topogin(name):
    def wrapper(cls):
        TOPOGIN_REGISTRY[name.lower()] = cls
        return cls
    return wrapper
