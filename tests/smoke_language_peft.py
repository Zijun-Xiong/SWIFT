"""CPU-only save/load smoke test for either bundled language PEFT copy."""

import argparse
import copy
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel


ROOT = Path(__file__).resolve().parents[1]


class ToyConfig(PretrainedConfig):
    model_type = "swift-toy"


class ToyModel(PreTrainedModel):
    config_class = ToyConfig

    def __init__(self, config):
        super().__init__(config)
        self.q_proj = nn.Linear(10, 11, bias=False)
        self.v_proj = nn.Linear(11, 7, bias=False)

    def forward(self, x):
        return self.v_proj(torch.tanh(self.q_proj(x)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        required=True,
        choices=["commonsense_reasoning", "natural_language_generation"],
    )
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / args.experiment / "peft" / "src"))

    from peft import LoraConfig, PeftModel, get_peft_model

    torch.manual_seed(11)
    base = ToyModel(ToyConfig())
    base_state = copy.deepcopy(base.state_dict())
    model = get_peft_model(
        base,
        LoraConfig(N=4, target_modules=["q_proj", "v_proj"]),
    )
    for module in model.modules():
        if hasattr(module, "lora_swift_blocks"):
            module.lora_swift_blocks.data.normal_()

    inputs = torch.randn(3, 10, requires_grad=True)
    expected = model(x=inputs)
    expected.sum().backward()
    trainable = [param for name, param in model.named_parameters() if "lora_swift_blocks" in name]
    assert trainable and all(param.grad is not None for param in trainable)

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir)
        fresh_base = ToyModel(ToyConfig())
        fresh_base.load_state_dict(base_state)
        restored = PeftModel.from_pretrained(fresh_base, tmpdir)
        restored.eval()
        actual = restored(x=inputs.detach())

    torch.testing.assert_close(actual, expected.detach())
    print(f"{args.experiment} PEFT save/load smoke test: PASS")


if __name__ == "__main__":
    main()
