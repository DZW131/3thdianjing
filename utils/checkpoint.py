import torch


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def state_dict_from_model(model):
    return unwrap_model(model).state_dict()


def _normalize_state_dict(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            normalized[key[len("module."):]] = value
        else:
            normalized[key] = value
    return normalized


def load_state_dict(model, state_dict, strict=True):
    unwrap_model(model).load_state_dict(_normalize_state_dict(state_dict), strict=strict)


def load_checkpoint(path, model, optimizer=None, map_location="cpu", strict=True):
    checkpoint = torch.load(path, map_location=map_location)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
        checkpoint = {"state_dict": state_dict}

    load_state_dict(model, state_dict, strict=strict)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint
