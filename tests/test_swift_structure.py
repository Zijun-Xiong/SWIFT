import ast
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATIONS = {
    "commonsense": ROOT / "commonsense_reasoning" / "peft" / "src" / "peft" / "tuners" / "lora.py",
    "generation": ROOT / "natural_language_generation" / "peft" / "src" / "peft" / "tuners" / "lora.py",
    "vision": ROOT / "image_classification" / "goat" / "peta" / "src" / "goat" / "layer.py",
}


def load_permutation_function(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_make_swift_permutation"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["_make_swift_permutation"]


def load_language_linear(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = {"_pad_to_multiple", "_make_swift_permutation", "LoraLayer", "Linear"}
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    namespace = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "transpose": lambda weight, fan_in_fan_out: weight.T if fan_in_fan_out else weight,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(nodes, type_ignores=[])), str(path), "exec"), namespace)
    return namespace["Linear"]


def load_vision_linear(path: Path):
    class BaseTunerLayer:
        def __init__(self):
            pass

        def _move_adapter_to_device_of_base_layer(self, adapter_name):
            self.to(self.base_layer.weight.device)

        @property
        def active_adapters(self):
            return [self._active_adapter]

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = {"_pad_to_multiple", "_make_swift_permutation", "GOATLayer", "LinearGOATLayer"}
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    namespace = {"torch": torch, "nn": nn, "F": F, "BaseTunerLayer": BaseTunerLayer}
    exec(compile(ast.fix_missing_locations(ast.Module(nodes, type_ignores=[])), str(path), "exec"), namespace)
    return namespace["LinearGOATLayer"]


class SwiftStructureTest(unittest.TestCase):
    def test_all_three_implementations_use_uniform_scattering(self):
        expected = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7])
        for name, path in IMPLEMENTATIONS.items():
            with self.subTest(name=name):
                permutation = load_permutation_function(path)(2, 4)
                self.assertTrue(torch.equal(permutation.cpu(), expected))

    def test_routing_is_deterministic_and_output_only(self):
        for name, path in IMPLEMENTATIONS.items():
            source = path.read_text(encoding="utf-8")
            with self.subTest(name=name):
                self.assertNotIn("randperm", source)
                self.assertNotIn("lora_perm_in", source)
                self.assertIn("index_select(-1, self.lora_swift_perm)", source)
                self.assertIn("persistent=True", source)
                self.assertIn("padded_in_features", source)
                self.assertIn("padded_out_features", source)

    def test_forward_backward_with_non_divisible_dimensions(self):
        for name in ("commonsense", "generation"):
            linear = load_language_linear(IMPLEMENTATIONS[name])(10, 11, N=4)
            linear.lora_swift_blocks.data.normal_()
            inputs = torch.randn(2, 3, 10, requires_grad=True)
            outputs = linear(inputs)
            outputs.sum().backward()
            self.assertEqual(outputs.shape, (2, 3, 11))
            self.assertIsNone(linear.weight.grad)
            self.assertIsNotNone(linear.lora_swift_blocks.grad)

        base_layer = nn.Linear(10, 11)
        linear = load_vision_linear(IMPLEMENTATIONS["vision"])(
            base_layer, "default", num_blocks=4
        )
        linear.lora_swift_blocks["default"].data.normal_()
        inputs = torch.randn(2, 3, 10, requires_grad=True)
        outputs = linear(inputs)
        outputs.sum().backward()
        self.assertEqual(outputs.shape, (2, 3, 11))
        self.assertIsNone(base_layer.weight.grad)
        self.assertIsNotNone(linear.lora_swift_blocks["default"].grad)

    def test_training_entrypoints_pass_the_block_count(self):
        commonsense = (ROOT / "commonsense_reasoning" / "finetune.py").read_text(encoding="utf-8")
        generation = (ROOT / "natural_language_generation" / "train.py").read_text(encoding="utf-8")
        vision_train = (ROOT / "image_classification" / "goat" / "train_vit.py").read_text(encoding="utf-8")
        vision_model = (
            ROOT / "image_classification" / "goat" / "peta" / "src" / "goat" / "model.py"
        ).read_text(encoding="utf-8")

        self.assertIn("N=swift_num_blocks", commonsense)
        self.assertIn("N=script_args.swift_num_blocks", generation)
        self.assertIn("swift_num_blocks=args.swift_num_blocks", vision_train)
        self.assertIn('"num_blocks": goat_config.swift_num_blocks', vision_model)

    def test_release_entrypoints_use_loadable_adapters(self):
        generation_train = (ROOT / "natural_language_generation" / "train.py").read_text(encoding="utf-8")
        math_eval = (
            ROOT / "natural_language_generation" / "scripts" / "math" / "evaluate_swift.sh"
        ).read_text(encoding="utf-8")
        code_eval = (
            ROOT / "natural_language_generation" / "scripts" / "code" / "evaluate_swift.sh"
        ).read_text(encoding="utf-8")
        vision_layer = IMPLEMENTATIONS["vision"].read_text(encoding="utf-8")
        vision_model = (
            ROOT / "image_classification" / "goat" / "peta" / "src" / "goat" / "model.py"
        ).read_text(encoding="utf-8")
        vision_train = (
            ROOT / "image_classification" / "goat" / "train_vit.py"
        ).read_text(encoding="utf-8")

        self.assertIn("resume_from_checkpoint=resume_from_checkpoint_dir", generation_train)
        self.assertIn('os.path.join(script_args.output_dir, "adapter_model")', generation_train)
        self.assertIn("--adapter_path", math_eval)
        self.assertIn("--adapter_path", code_eval)
        self.assertFalse((ROOT / "natural_language_generation" / "utils" / "gen_vllm.py").exists())
        self.assertIn('adapter_layer_names = ("lora_swift_blocks",)', vision_layer)
        self.assertIn("module.enable_adapters(enabled)", vision_model)
        self.assertIn("save_swift_adapter", vision_train)

    def test_python_sources_parse(self):
        for path in ROOT.rglob("*.py"):
            with self.subTest(path=str(path.relative_to(ROOT))):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


if __name__ == "__main__":
    unittest.main()

