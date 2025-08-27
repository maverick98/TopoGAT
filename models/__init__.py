# models/__init__.py
# 🔁 Ensure decorators in these files run and register models
from models import topogat  # <-- this will register TopoGAT variants
from models import topogin  # <-- this will register TopoGIN variants

from models.topogat_selector import get_topogat_model
from models.topogin_selector import get_topogin_model

SUPPORTED_MODELS = ("topogat", "topogin")

def get_model(name, variant="basic", **kwargs):
    name = name.lower()
    if name.startswith("topogat"):
        return get_topogat_model(variant=variant, **kwargs)
    elif name.startswith("topogin"):
        return get_topogin_model(variant=variant, **kwargs)
    else:
        raise ValueError(f"Unknown model type '{name}'. Supported: {SUPPORTED_MODELS}")


