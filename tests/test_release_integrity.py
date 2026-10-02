import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {
    "commonsense_reasoning": {
        "name": "swift-commonsense",
        "conda": {"python=3.10", "pytorch=2.1.0", "pytorch-cuda=12.1"},
    },
    "natural_language_generation": {
        "name": "swift-nlg",
        "conda": {
            "python=3.10",
            "pytorch=2.4.0",
            "torchvision=0.19.0",
            "pytorch-cuda=12.1",
            "nvidia/label/cuda-12.1.0::cuda-toolkit=12.1.0",
        },
    },
    "image_classification": {
        "name": "swift-vision",
        "conda": {
            "python=3.9",
            "pytorch=2.4.0",
            "torchvision=0.19.0",
            "pytorch-cuda=12.1",
            "nvidia/label/cuda-12.1.0::cuda-toolkit=12.1.0",
        },
    },
}


def requirement_lines(path: Path):
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def environment_pip_lines(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    pip_index = lines.index("  - pip:")
    return {line.strip()[2:] for line in lines[pip_index + 1 :] if line.startswith("      - ")}


class ReleaseIntegrityTest(unittest.TestCase):
    def test_environment_files_match_requirements(self):
        for experiment, expected in EXPERIMENTS.items():
            with self.subTest(experiment=experiment):
                directory = ROOT / experiment
                environment_path = directory / "environment.yml"
                environment_text = environment_path.read_text(encoding="utf-8")
                self.assertIn(f"name: {expected['name']}", environment_text)
                for dependency in expected["conda"]:
                    self.assertIn(f"  - {dependency}\n", environment_text)
                self.assertEqual(
                    requirement_lines(directory / "requirements.txt"),
                    environment_pip_lines(environment_path),
                )

    def test_launchers_are_strict_and_have_preflight_checks(self):
        scripts = list(ROOT.glob("*/scripts/**/*.sh"))
        self.assertEqual(len(scripts), 9)
        for script in scripts:
            with self.subTest(script=str(script.relative_to(ROOT))):
                source = script.read_text(encoding="utf-8")
                self.assertIn("set -euo pipefail", source)

        self.assertIn(
            "Training data not found",
            (ROOT / "commonsense_reasoning/scripts/train_swift.sh").read_text(encoding="utf-8"),
        )
        self.assertIn(
            "metamath/train.json",
            (ROOT / "natural_language_generation/scripts/math/train_swift.sh").read_text(encoding="utf-8"),
        )
        self.assertIn(
            "python/train.json",
            (ROOT / "natural_language_generation/scripts/code/train_swift.sh").read_text(encoding="utf-8"),
        )
        self.assertIn(
            "download_data.py",
            (ROOT / "image_classification/scripts/train_swift.sh").read_text(encoding="utf-8"),
        )

    def test_release_has_no_machine_specific_paths_or_large_artifacts(self):
        forbidden = ("F:" + "\\", "C:" + "\\Users\\")
        for path in ROOT.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            with self.subTest(path=str(path.relative_to(ROOT))):
                self.assertLess(path.stat().st_size, 5 * 1024 * 1024)
                if path.suffix.lower() in {".py", ".sh", ".md", ".txt", ".yml", ".json"}:
                    source = path.read_text(encoding="utf-8")
                    self.assertFalse(any(value in source for value in forbidden))


if __name__ == "__main__":
    unittest.main()
