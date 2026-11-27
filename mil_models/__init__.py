import torch

from .abmil import ABMIL
from .dtfd import DTFDMIL
from .ilra import ILRA
from .transmil import TransMIL
from .wikg import WiKG

ARCHITECTURES = {"abmil": ABMIL, "dtfd": DTFDMIL, "transmil": TransMIL, "wikg": WiKG, "ilra": ILRA}


def build_model(name, in_dim, num_classes, **kwargs):
    return ARCHITECTURES[name](in_dim, num_classes, **kwargs)


def load_prior(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return model.load_state_dict(checkpoint["state_dict"], strict=False)
