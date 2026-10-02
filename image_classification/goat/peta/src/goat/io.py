import os
from pathlib import Path

import torch
from peft import get_peft_model
from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING
from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
from peft.utils import WEIGHTS_NAME

from ..utils.save_and_load import get_peft_model_state_dict, set_peft_model_state_dict
from .config import GOATConfig
from .model import GOATModel


def register_swift_with_peft():
    """Register the custom SWIFT config and tuner in PEFT 0.14."""
    PEFT_TYPE_TO_CONFIG_MAPPING["GOAT"] = GOATConfig
    PEFT_TYPE_TO_MODEL_MAPPING["GOAT"] = GOATModel


def save_swift_adapter(model, save_directory, adapter_name="default"):
    """Save SWIFT blocks and modules_to_save with the custom serializer."""
    save_directory = Path(save_directory)
    save_directory.mkdir(parents=True, exist_ok=True)
    state_dict = get_peft_model_state_dict(model, adapter_name=adapter_name)
    cpu_state_dict = {key: value.detach().cpu() for key, value in state_dict.items()}
    torch.save(cpu_state_dict, save_directory / WEIGHTS_NAME)

    config = model.peft_config[adapter_name]
    inference_mode = config.inference_mode
    config.inference_mode = True
    config.save_pretrained(save_directory)
    config.inference_mode = inference_mode


def load_swift_adapter(model, adapter_directory, adapter_name="default", is_trainable=False):
    """Attach a saved SWIFT adapter to an already loaded base model."""
    adapter_directory = os.fspath(adapter_directory)
    weights_path = os.path.join(adapter_directory, WEIGHTS_NAME)
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"SWIFT weights not found: {weights_path}")

    register_swift_with_peft()
    config = GOATConfig.from_pretrained(adapter_directory)
    config.inference_mode = not is_trainable
    wrapped_model = get_peft_model(model, config, adapter_name=adapter_name)
    state_dict = torch.load(weights_path, map_location="cpu")
    load_result = set_peft_model_state_dict(wrapped_model, state_dict, adapter_name=adapter_name)
    if load_result.unexpected_keys:
        raise RuntimeError(f"Unexpected keys while loading SWIFT: {load_result.unexpected_keys}")
    if not is_trainable:
        wrapped_model.eval()
    return wrapped_model
