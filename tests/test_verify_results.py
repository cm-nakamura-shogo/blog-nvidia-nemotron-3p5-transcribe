import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from verify_results import expected_combinations, main


class VerifyResultsTest(unittest.TestCase):
    model_ids = [f"model-{index}" for index in range(7)]

    def test_full_performance_matrix_has_126_runs(self):
        runs = expected_combinations(
            self.model_ids,
            [1, 4, 8, 16, 32, 64],
            3,
        )
        self.assertEqual(len(runs), 126)

    def test_accuracy_override_has_seven_bs1_runs(self):
        runs = expected_combinations(self.model_ids, [1], 1)
        self.assertEqual(len(runs), 7)
        self.assertEqual({batch_size for _, batch_size, _ in runs}, {1})
        self.assertEqual({repeat for _, _, repeat in runs}, {1})

    def test_available_only_accepts_a_partial_result_without_bs1(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_dir = root / "results"
            input_dir.mkdir()
            prediction_path = input_dir / "model-a-bs8-run2.jsonl"
            prediction_path.touch()
            metadata_path = input_dir / "model-a-bs8-run2.meta.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "model_id": "model-a",
                        "model_revision": "revision-a",
                        "batch_size": 8,
                        "repeat": 2,
                        "predictions": str(prediction_path),
                    }
                ),
                encoding="utf-8",
            )
            output_path = root / "verification.json"
            manifest_rows = [
                {
                    "sample_id": "sample-1",
                    "audio_filepath": "audio-1.wav",
                    "duration_sec": 1.0,
                    "reference": "一",
                },
                {
                    "sample_id": "sample-2",
                    "audio_filepath": "audio-2.wav",
                    "duration_sec": 1.0,
                    "reference": "二",
                },
            ]
            prediction_rows = [
                {"sample_id": "sample-1"},
                {"sample_id": "sample-2"},
            ]

            def fake_load_yaml(path: Path):
                if path == Path("matrix.yaml"):
                    return {
                        "manifest": "manifest.jsonl",
                        "batch_sizes": [1, 4],
                        "repeats": 3,
                        "configs": ["model.yaml"],
                    }
                return {"id": "model-a", "model_revision": "revision-a"}

            def fake_read_jsonl(path: Path):
                if path == Path("manifest.jsonl"):
                    return manifest_rows
                if path == prediction_path:
                    return prediction_rows
                raise AssertionError(f"Unexpected JSONL path: {path}")

            argv = [
                "verify_results.py",
                "--matrix",
                "matrix.yaml",
                "--input-dir",
                str(input_dir),
                "--available-only",
                "--output",
                str(output_path),
            ]
            with (
                patch("verify_results.load_yaml", side_effect=fake_load_yaml),
                patch("verify_results.read_jsonl", side_effect=fake_read_jsonl),
                patch.object(sys, "argv", argv),
            ):
                main()

            result = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(result["mode"], "available_only")
            self.assertEqual(result["verified_runs"], 1)
            self.assertEqual(result["batch_sizes"], [8])
            self.assertEqual(result["repeats"], [2])


if __name__ == "__main__":
    unittest.main()
