import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from common import benchmark_summary, ordered_rows, percentile, validate_manifest


class CommonTest(unittest.TestCase):
    def test_bucket_order_is_deterministic(self):
        rows = [
            {"sample_id": "b", "duration_sec": 1.0},
            {"sample_id": "c", "duration_sec": 2.0},
            {"sample_id": "a", "duration_sec": 1.0},
        ]
        self.assertEqual([row["sample_id"] for row in ordered_rows(rows, "bucket")], ["c", "a", "b"])

    def test_percentile_interpolates(self):
        self.assertEqual(percentile([1.0, 2.0, 3.0], 0.5), 2.0)

    def test_benchmark_summary(self):
        summary = benchmark_summary(
            total_files=10,
            total_audio_sec=100.0,
            end_to_end_elapsed_sec=2.0,
            gpu_inference_elapsed_sec=1.0,
        )
        self.assertEqual(summary["rtf"], 0.02)
        self.assertEqual(summary["audio_hours_per_hour"], 50.0)
        self.assertEqual(summary["files_per_sec"], 5.0)

    def test_manifest_rejects_duplicate_sample_ids(self):
        rows = [
            {"sample_id": "same", "audio_filepath": "a.wav", "duration_sec": 1, "reference": "a"},
            {"sample_id": "same", "audio_filepath": "b.wav", "duration_sec": 1, "reference": "b"},
        ]
        with self.assertRaisesRegex(ValueError, "sample_id"):
            validate_manifest(rows)

    def test_manifest_rejects_duplicate_audio_paths(self):
        rows = [
            {"sample_id": "a", "audio_filepath": "same.wav", "duration_sec": 1, "reference": "a"},
            {"sample_id": "b", "audio_filepath": "same.wav", "duration_sec": 1, "reference": "b"},
        ]
        with self.assertRaisesRegex(ValueError, "audio_filepath"):
            validate_manifest(rows)


if __name__ == "__main__":
    unittest.main()
