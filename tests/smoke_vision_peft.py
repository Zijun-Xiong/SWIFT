"""CPU-only PEFT 0.14 smoke test for the vision SWIFT adapter."""

import copy
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from peft import get_peft_model
from transformers import PretrainedConfig, PreTrainedModel


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "image_classification" / "goat"))

import peta.src  # noqa: E402


class ToyConfig(PretrainedConfig):
    model_type = "swift-toy"


class ToyModel(PreTrainedModel):
    config_class = ToyConfig

    def __init__(self, config):
        super().__init__(config)
        self.q_proj = nn.Linear(10, 11, bias=False)
        self.classifier = nn.Linear(11, 3)

    def forward(self, x):
        return self.classifier(torch.tanh(self.q_proj(x)))


def main():
    torch.manual_seed(7)
    base = ToyModel(ToyConfig())
    base_state = copy.deepcopy(base.state_dict())
    config = peta.src.GOATConfig(
        swift_num_blocks=4,
        target_modules=["q_proj"],
        modules_to_save=["classifier"],
        bias="none",
    )
    peta.src.register_swift_with_peft()
    model = get_peft_model(base, config)
    swift_layer = model.base_model.model.q_proj
    swift_layer.lora_swift_blocks["default"].data.normal_()

    inputs = torch.randn(2, 10, requires_grad=True)
    expected = model(x=inputs)
    expected.sum().backward()
    assert swift_layer.lora_swift_blocks["default"].grad is not None
    assert swift_layer.base_layer.weight.grad is None

    with tempfile.TemporaryDirectory() as tmpdir:
        peta.src.save_swift_adapter(model, tmpdir)
        fresh_base = ToyModel(ToyConfig())
        fresh_base.load_state_dict(base_state)
        restored = peta.src.load_swift_adapter(fresh_base, tmpdir)
        actual = restored(x=inputs.detach())

    torch.testing.assert_close(actual, expected.detach())
    print("vision PEFT save/load smoke test: PASS")


if __name__ == "__main__":
    main()
