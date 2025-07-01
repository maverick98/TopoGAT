from models.topogat import BaseTopoGAT
from models.gat import BaseGAT
from models.gin import BaseGIN

MODEL_REGISTRY = {
    "gat": BaseGAT,
    "gin": BaseGAT,
    "topogat": BaseTopoGAT
}

def get_model(name, **kwargs):
    return MODEL_REGISTRY[name](**kwargs)
