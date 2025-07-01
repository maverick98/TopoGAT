MODEL_REGISTRY = {}

def register_topogat(name):
    def wrapper(cls):
        MODEL_REGISTRY[name.lower()] = cls
        return cls
    return wrapper
