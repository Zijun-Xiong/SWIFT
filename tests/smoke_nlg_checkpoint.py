"""CPU-only checkpoint/resume smoke test for the generation trainer."""

import copy
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import PretrainedConfig, PreTrainedModel


ROOT = Path(__file__).resolve().parents[1]
NLG_ROOT = ROOT / "natural_language_generation"
sys.path.insert(0, str(NLG_ROOT))

from train import SwiftTrainer, TrainingArguments, get_last_checkpoint  # noqa: E402
from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402


class ToyConfig(PretrainedConfig):
    model_type = "swift-toy"


class ToyModel(PreTrainedModel):
    config_class = ToyConfig

    def __init__(self, config):
        super().__init__(config)
        self.q_proj = nn.Linear(10, 11, bias=False)
        self.v_proj = nn.Linear(11, 7, bias=False)

    def forward(self, x, labels=None):
        logits = self.v_proj(torch.tanh(self.q_proj(x)))
        loss = None if labels is None else nn.functional.mse_loss(logits, labels)
        return {"loss": loss, "logits": logits}


class ToyDataset(Dataset):
    def __init__(self):
        generator = torch.Generator().manual_seed(19)
        self.x = torch.randn(8, 10, generator=generator)
        self.labels = torch.randn(8, 7, generator=generator)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return {"x": self.x[index], "labels": self.labels[index]}


def make_model(base_state):
    base = ToyModel(ToyConfig())
    base.load_state_dict(base_state)
    return get_peft_model(
        base,
        LoraConfig(N=4, target_modules=["q_proj", "v_proj"]),
    )


def make_args(output_dir, max_steps):
    return TrainingArguments(
        output_dir=output_dir,
        full_finetune=False,
        max_steps=max_steps,
        per_device_train_batch_size=2,
        learning_rate=1e-3,
        logging_strategy="no",
        save_strategy="steps",
        save_steps=1,
        save_total_limit=2,
        report_to=[],
        disable_tqdm=True,
        remove_unused_columns=False,
    )


def main():
    torch.manual_seed(17)
    initial = ToyModel(ToyConfig())
    base_state = copy.deepcopy(initial.state_dict())
    dataset = ToyDataset()

    with tempfile.TemporaryDirectory() as tmpdir:
        first = SwiftTrainer(
            model=make_model(base_state),
            args=make_args(tmpdir, max_steps=2),
            train_dataset=dataset,
        )
        first.train()
        checkpoint = get_last_checkpoint(tmpdir)
        assert checkpoint is not None
        adapter_dir = Path(checkpoint) / "adapter_model"
        assert (adapter_dir / "adapter_model.bin").is_file()
        assert not (Path(checkpoint) / "model.safetensors").exists()

        fresh_base = ToyModel(ToyConfig())
        fresh_base.load_state_dict(base_state)
        resumed_model = PeftModel.from_pretrained(fresh_base, str(adapter_dir), is_trainable=True)
        resumed = SwiftTrainer(
            model=resumed_model,
            args=make_args(tmpdir, max_steps=3),
            train_dataset=dataset,
        )
        result = resumed.train(resume_from_checkpoint=checkpoint)
        assert result.global_step == 3

    print("generation Trainer checkpoint/resume smoke test: PASS")


if __name__ == "__main__":
    main()
