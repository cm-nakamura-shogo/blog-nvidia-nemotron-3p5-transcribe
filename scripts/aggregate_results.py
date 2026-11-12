#!/usr/bin/env python3
"""Aggregate repeated runs, extract worst cases, and create benchmark figures."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from calculate_cer import has_suspicious_repetition, score_rows
from common import read_jsonl, write_json


METRICS = [
    "end_to_end_elapsed_sec",
    "gpu_inference_elapsed_sec",
    "rtf",
    "audio_hours_per_hour",
    "files_per_sec",
    "peak_vram_mib_nvidia_smi",
    "peak_allocated_mib_torch",
    "peak_reserved_mib_torch",
]

DISPLAY_NAMES = {
    "faster-whisper-large-v3": "Whisper large-v3",
    "faster-whisper-large-v3-turbo": "Whisper large-v3-turbo",
    "kotoba-whisper-v2.2": "Kotoba-Whisper v2.2",
    "parakeet-tdt-ctc-0.6b-ja": "Parakeet TDT 0.6B Japanese",
    "reazonspeech-k2-v2": "ReazonSpeech K2 v2",
    "nemotron-3.5-asr-offline-transformers": "Nemotron 3.5 / Transformers (offline)",
    "nemotron-3.5-asr-streaming-0.6b": "Nemotron 3.5 / NeMo (streaming, ref.)",
    "nemotron-3.5-transcribe-fp8-vllm": "Nemotron 3.5 Transcribe FP8 / vLLM",
}

FIGURE_STYLE = {
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "legend.title_fontsize": 11,
}


def apply_figure_style() -> None:
    """Use text sizes that remain readable when figures are scaled to blog width."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(FIGURE_STYLE)


def median_or_none(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return statistics.median(numeric) if numeric else None


def mean_or_none(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return statistics.fmean(numeric) if numeric else None


def pstdev_or_none(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return statistics.pstdev(numeric) if numeric else None


def load_metadata(
    input_dir: Path, *, include_accuracy: bool
) -> list[dict[str, Any]]:
    metadata = []
    for path in sorted(input_dir.rglob("*.meta.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        value["metadata_path"] = str(path)
        if include_accuracy:
            prediction_path = Path(value["predictions"])
            if not prediction_path.exists():
                prediction_path = path.parent / prediction_path.name
            if not prediction_path.exists():
                raise FileNotFoundError(
                    f"Prediction file not found for accuracy run: {path}"
                )
            _, value["accuracy"] = score_rows(read_jsonl(prediction_path))
        metadata.append(value)
    if not metadata:
        raise ValueError(f"No *.meta.json files found under {input_dir}")
    return metadata


def aggregate_by_batch(metadata: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    keys = sorted({(item["model_id"], int(item["batch_size"])) for item in metadata})
    for model_id, batch_size in keys:
        runs = [
            item
            for item in metadata
            if item["model_id"] == model_id and int(item["batch_size"]) == batch_size
        ]
        row: dict[str, Any] = {
            "model_id": model_id,
            "batch_size": batch_size,
            "runs": len(runs),
            "precision": runs[0].get("precision"),
            "runtime": runs[0].get("runtime"),
            "comparison_scope": runs[0].get("comparison_scope")
            or (
                "streaming_reference"
                if model_id == "nemotron-3.5-asr-streaming-0.6b"
                else "main"
            ),
        }
        for metric in METRICS:
            row[metric] = median_or_none([run.get(metric) for run in runs])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["model_id", "batch_size"])


def make_accuracy_summary(metadata: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    metric_names = [
        "raw_cer",
        "normalized_cer",
        "exact_match_rate",
        "empty_output_rate",
        "suspicious_repetition_count",
        "normalized_substitutions",
        "normalized_deletions",
        "normalized_insertions",
    ]
    for model_id in sorted({item["model_id"] for item in metadata}):
        runs = [item for item in metadata if item["model_id"] == model_id]
        row: dict[str, Any] = {
            "model_id": model_id,
            "display_name": DISPLAY_NAMES.get(model_id, model_id),
            "runtime": runs[0].get("runtime"),
            "precision": runs[0].get("precision"),
            "comparison_scope": runs[0].get("comparison_scope")
            or (
                "streaming_reference"
                if model_id == "nemotron-3.5-asr-streaming-0.6b"
                else "main"
            ),
            "runs": len(runs),
            "batch_sizes": sorted({int(item["batch_size"]) for item in runs}),
            "repeat_ids": sorted({int(item["repeat"]) for item in runs}),
        }
        for metric_name in metric_names:
            if metric_name.startswith("normalized_") and metric_name not in {
                "normalized_cer"
            }:
                values = [
                    item["accuracy"].get("normalized_counts", {}).get(
                        metric_name.removeprefix("normalized_")
                    )
                    for item in runs
                ]
            else:
                values = [item["accuracy"].get(metric_name) for item in runs]
            row[metric_name] = mean_or_none(values)
            if metric_name in {"raw_cer", "normalized_cer"}:
                numeric = [float(value) for value in values if value is not None]
                row[f"{metric_name}_min"] = min(numeric) if numeric else None
                row[f"{metric_name}_max"] = max(numeric) if numeric else None
                row[f"{metric_name}_std"] = pstdev_or_none(values)
        for threshold in (20, 50):
            key = f"normalized_cer_ge_{threshold}pct_rate"
            row[key] = mean_or_none(
                [
                    item["accuracy"]
                    .get(f"normalized_cer_ge_{threshold}pct", {})
                    .get("rate")
                    for item in runs
                ]
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("normalized_cer").reset_index(drop=True)


def make_performance_summary(performance_by_batch: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_id, group in performance_by_batch.groupby("model_id"):
        best_row = group.loc[group["audio_hours_per_hour"].idxmax()]
        rows.append(
            {
                "model_id": model_id,
                "display_name": DISPLAY_NAMES.get(model_id, model_id),
                "runtime": best_row["runtime"],
                "precision": best_row["precision"],
                "comparison_scope": best_row["comparison_scope"],
                "runs": int(group["runs"].sum()),
                "batch_sizes": sorted(int(value) for value in group["batch_size"]),
                "best_batch_size": int(best_row["batch_size"]),
                "best_batch_runs": int(best_row["runs"]),
                "audio_hours_per_hour": best_row["audio_hours_per_hour"],
                "rtf": best_row["rtf"],
                "files_per_sec": best_row["files_per_sec"],
                "peak_vram_mib_nvidia_smi": best_row["peak_vram_mib_nvidia_smi"],
            }
        )
    return pd.DataFrame(rows).sort_values("audio_hours_per_hour", ascending=False).reset_index(
        drop=True
    )


def make_model_summary(
    accuracy_summary: pd.DataFrame,
    performance_summary: pd.DataFrame,
) -> pd.DataFrame:
    accuracy_models = set(accuracy_summary["model_id"])
    performance_models = set(performance_summary["model_id"])
    shared_models = accuracy_models & performance_models
    if not shared_models:
        raise ValueError("Accuracy and performance inputs have no models in common")

    rows = []
    for model_id in sorted(shared_models):
        accuracy_row = accuracy_summary[accuracy_summary["model_id"] == model_id].iloc[0]
        best_row = performance_summary[performance_summary["model_id"] == model_id].iloc[0]
        rows.append(
            {
                "model_id": model_id,
                "display_name": DISPLAY_NAMES.get(model_id, model_id),
                "runtime": accuracy_row["runtime"],
                "precision": accuracy_row["precision"],
                "comparison_scope": accuracy_row["comparison_scope"],
                "accuracy_runs": int(accuracy_row["runs"]),
                "accuracy_batch_sizes": accuracy_row["batch_sizes"],
                "raw_cer": accuracy_row["raw_cer"],
                "normalized_cer": accuracy_row["normalized_cer"],
                "normalized_cer_min": accuracy_row["normalized_cer_min"],
                "normalized_cer_max": accuracy_row["normalized_cer_max"],
                "normalized_cer_std": accuracy_row["normalized_cer_std"],
                "exact_match_rate": accuracy_row["exact_match_rate"],
                "empty_output_rate": accuracy_row["empty_output_rate"],
                "normalized_cer_ge_20pct_rate": accuracy_row[
                    "normalized_cer_ge_20pct_rate"
                ],
                "normalized_cer_ge_50pct_rate": accuracy_row[
                    "normalized_cer_ge_50pct_rate"
                ],
                "suspicious_repetition_count": accuracy_row["suspicious_repetition_count"],
                "normalized_substitutions": accuracy_row["normalized_substitutions"],
                "normalized_deletions": accuracy_row["normalized_deletions"],
                "normalized_insertions": accuracy_row["normalized_insertions"],
                "performance_runs": int(best_row["runs"]),
                "performance_batch_sizes": best_row["batch_sizes"],
                "best_batch_size": int(best_row["best_batch_size"]),
                "best_batch_runs": int(best_row["best_batch_runs"]),
                "audio_hours_per_hour": best_row["audio_hours_per_hour"],
                "rtf": best_row["rtf"],
                "files_per_sec": best_row["files_per_sec"],
                "peak_vram_mib_nvidia_smi": best_row["peak_vram_mib_nvidia_smi"],
            }
        )
    summary = pd.DataFrame(rows).sort_values("normalized_cer").reset_index(drop=True)
    frontier = []
    for index, row in summary.iterrows():
        if row["comparison_scope"] != "main":
            frontier.append(False)
            continue
        dominated = False
        for other_index, other in summary.iterrows():
            if index == other_index:
                continue
            if other["comparison_scope"] != "main":
                continue
            no_worse = (
                other["normalized_cer"] <= row["normalized_cer"]
                and other["audio_hours_per_hour"] >= row["audio_hours_per_hour"]
            )
            strictly_better = (
                other["normalized_cer"] < row["normalized_cer"]
                or other["audio_hours_per_hour"] > row["audio_hours_per_hour"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        frontier.append(not dominated)
    summary["pareto_frontier"] = frontier
    return summary


def select_representative_run(
    metadata: list[dict[str, Any]], model_id: str
) -> dict[str, Any]:
    """Select the lowest available batch size and repeat for qualitative examples."""

    candidates = [item for item in metadata if item["model_id"] == model_id]
    if not candidates:
        raise ValueError(f"No available run for {model_id}")
    return sorted(
        candidates,
        key=lambda item: (int(item["batch_size"]), int(item["repeat"])),
    )[0]


def save_accuracy_figure(summary: pd.DataFrame, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    apply_figure_style()
    summary = summary.assign(
        display_name=summary["model_id"].map(DISPLAY_NAMES).fillna(summary["model_id"])
    )
    color_map = plt.get_cmap("tab10")
    model_colors = {
        model_id: color_map(index)
        for index, model_id in enumerate(DISPLAY_NAMES)
    }

    cer_percent = summary.assign(normalized_cer_percent=summary["normalized_cer"] * 100)
    ax = cer_percent.plot.bar(
        x="display_name",
        y="normalized_cer_percent",
        color=[model_colors[model_id] for model_id in cer_percent["model_id"]],
        legend=False,
        figsize=(12, 6),
    )
    ax.bar_label(ax.containers[0], fmt="%.2f%%", padding=3, fontsize=11)
    ax.set_ylabel("Normalized CER (%)")
    ax.set_xlabel("")
    ax.set_title("FLEURS ja_jp test — NVIDIA L40S")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(figure_dir / "normalized-cer.png", dpi=180)
    plt.close()


def save_figures(by_batch: pd.DataFrame, summary: pd.DataFrame, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    apply_figure_style()
    summary = summary.assign(
        display_name=summary["model_id"].map(DISPLAY_NAMES).fillna(summary["model_id"])
    )
    color_map = plt.get_cmap("tab10")
    model_colors = {
        model_id: color_map(index)
        for index, model_id in enumerate(DISPLAY_NAMES)
    }

    if "normalized_cer" in summary.columns:
        save_accuracy_figure(summary, figure_dir)

    ax = summary.plot.bar(
        x="display_name",
        y="audio_hours_per_hour",
        color=[model_colors[model_id] for model_id in summary["model_id"]],
        legend=False,
        figsize=(12, 6),
    )
    ax.bar_label(ax.containers[0], fmt="%.1f", padding=3, fontsize=11)
    ax.set_ylabel("Audio hours / wall-clock hour")
    ax.set_xlabel("")
    ax.set_title("Best warm batch throughput — NVIDIA L40S")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(figure_dir / "audio-throughput.png", dpi=180)
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 6))
    for model_id, group in by_batch.groupby("model_id"):
        is_reference = group.iloc[0]["comparison_scope"] != "main"
        ax.plot(
            group["batch_size"],
            group["audio_hours_per_hour"],
            marker="D" if is_reference else "o",
            linestyle="--" if is_reference else "-",
            alpha=0.75 if is_reference else 1.0,
            color=model_colors[model_id],
            label=DISPLAY_NAMES.get(model_id, model_id),
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(by_batch["batch_size"].unique()))
    ax.set_xticklabels([str(value) for value in sorted(by_batch["batch_size"].unique())])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Audio hours / wall-clock hour")
    ax.set_title("Batch scaling — FLEURS ja_jp test, NVIDIA L40S")
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(figure_dir / "batch-size-vs-throughput.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    for model_id, group in by_batch.groupby("model_id"):
        is_reference = group.iloc[0]["comparison_scope"] != "main"
        ax.plot(
            group["batch_size"],
            group["peak_vram_mib_nvidia_smi"] / 1024,
            marker="D" if is_reference else "o",
            linestyle="--" if is_reference else "-",
            alpha=0.75 if is_reference else 1.0,
            color=model_colors[model_id],
            label=DISPLAY_NAMES.get(model_id, model_id),
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(by_batch["batch_size"].unique()))
    ax.set_xticklabels([str(value) for value in sorted(by_batch["batch_size"].unique())])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Peak VRAM (GiB, nvidia-smi)")
    ax.set_title("VRAM scaling — FLEURS ja_jp test, NVIDIA L40S")
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(figure_dir / "batch-size-vs-vram.png", dpi=180)
    plt.close(fig)

    if "normalized_cer" not in summary.columns:
        return

    fig, ax = plt.subplots(figsize=(13, 7))
    minimum_vram = float(summary["peak_vram_mib_nvidia_smi"].min())
    maximum_vram = float(summary["peak_vram_mib_nvidia_smi"].max())
    legend_handles = []
    for _, row in summary.iterrows():
        vram = float(row["peak_vram_mib_nvidia_smi"])
        normalized_vram = (vram - minimum_vram) / max(1.0, maximum_vram - minimum_vram)
        size = 140 + 420 * normalized_vram
        is_reference = row["comparison_scope"] != "main"
        marker = "X" if is_reference else "o"
        ax.scatter(
            row["normalized_cer"] * 100,
            row["audio_hours_per_hour"],
            s=size,
            marker=marker,
            alpha=0.65 if is_reference else 0.8,
            color=model_colors[row["model_id"]],
            edgecolor="black" if row["pareto_frontier"] else "dimgray",
            linewidth=1.5,
        )
        ax.annotate(
            row["display_name"],
            (row["normalized_cer"] * 100, row["audio_hours_per_hour"]),
            xytext=(-8, 5) if row["normalized_cer"] * 100 > 12 else (6, 5),
            textcoords="offset points",
            ha="right" if row["normalized_cer"] * 100 > 12 else "left",
            fontsize=11,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color="none",
                markerfacecolor=model_colors[row["model_id"]],
                markeredgecolor="black" if row["pareto_frontier"] else "dimgray",
                markersize=8,
                label=row["display_name"],
            )
        )
    ax.set_xlabel("Normalized CER (%) — lower is better")
    ax.set_ylabel("Audio hours / wall-clock hour (higher is better)")
    ax.set_title("Accuracy–throughput trade-off — NVIDIA L40S")
    ax.legend(
        handles=legend_handles,
        fontsize=11,
        title="Model / runtime",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "pareto.png", dpi=180)
    plt.close(fig)


def extract_worst_cases(metadata: list[dict[str, Any]], output_dir: Path) -> None:
    rows = []
    for model_id in sorted({item["model_id"] for item in metadata}):
        selected = select_representative_run(metadata, model_id)
        prediction_path = Path(selected["predictions"])
        if not prediction_path.exists():
            metadata_path = Path(selected["metadata_path"])
            prediction_path = metadata_path.parent / prediction_path.name
        scored_rows, _ = score_rows(read_jsonl(prediction_path))
        details = sorted(
            scored_rows,
            key=lambda item: item["normalized_cer"],
            reverse=True,
        )[:10]
        for rank, detail in enumerate(details, start=1):
            rows.append(
                {
                    "model_id": model_id,
                    "batch_size": int(selected["batch_size"]),
                    "repeat": int(selected["repeat"]),
                    "rank": rank,
                    "sample_id": detail["sample_id"],
                    "normalized_cer": detail["normalized_cer"],
                    "raw_cer": detail["raw_cer"],
                    "reference": detail["reference"],
                    "hypothesis": detail["hypothesis"],
                    "suspicious_repetition": has_suspicious_repetition(detail["hypothesis"]),
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "worst-cases.csv", index=False)


def extract_contrasts(metadata: list[dict[str, Any]], output_dir: Path) -> None:
    """Create reproducible candidate sets for the article's qualitative review."""

    runs: dict[str, dict[str, dict[str, Any]]] = {}
    for model_id in sorted({item["model_id"] for item in metadata}):
        selected = select_representative_run(metadata, model_id)
        prediction_path = Path(selected["predictions"])
        if not prediction_path.exists():
            prediction_path = Path(selected["metadata_path"]).parent / prediction_path.name
        details, _ = score_rows(read_jsonl(prediction_path))
        runs[model_id] = {row["sample_id"]: row for row in details}

    required = {
        "faster-whisper-large-v3",
        "faster-whisper-large-v3-turbo",
        "kotoba-whisper-v2.2",
        "nemotron-3.5-asr-streaming-0.6b",
        "nemotron-3.5-asr-offline-transformers",
        "parakeet-tdt-ctc-0.6b-ja",
        "reazonspeech-k2-v2",
    }
    if not required.issubset(runs):
        return

    large = runs["faster-whisper-large-v3"]
    turbo = runs["faster-whisper-large-v3-turbo"]
    kotoba = runs["kotoba-whisper-v2.2"]
    nemotron_streaming = runs["nemotron-3.5-asr-streaming-0.6b"]
    nemotron_offline = runs["nemotron-3.5-asr-offline-transformers"]
    parakeet = runs["parakeet-tdt-ctc-0.6b-ja"]
    reazonspeech = runs["reazonspeech-k2-v2"]
    model_rows = {
        "faster-whisper-large-v3": large,
        "faster-whisper-large-v3-turbo": turbo,
        "kotoba-whisper-v2.2": kotoba,
        "nemotron-3.5-asr-streaming-0.6b": nemotron_streaming,
        "nemotron-3.5-asr-offline-transformers": nemotron_offline,
        "parakeet-tdt-ctc-0.6b-ja": parakeet,
        "reazonspeech-k2-v2": reazonspeech,
    }
    short_names = {
        "faster-whisper-large-v3": "large-v3",
        "faster-whisper-large-v3-turbo": "large-v3-turbo",
        "kotoba-whisper-v2.2": "kotoba",
        "nemotron-3.5-asr-streaming-0.6b": "nemotron_streaming",
        "nemotron-3.5-asr-offline-transformers": "nemotron_offline",
        "parakeet-tdt-ctc-0.6b-ja": "parakeet",
        "reazonspeech-k2-v2": "reazonspeech",
    }
    candidates: list[dict[str, Any]] = []

    def add(category: str, sample_id: str, score: float) -> None:
        base = large[sample_id]
        row: dict[str, Any] = {
            "category": category,
            "sample_id": sample_id,
            "score": score,
            "reference": base["reference"],
        }
        for model_id, by_sample in model_rows.items():
            short = short_names[model_id]
            row[f"{short}_cer"] = by_sample[sample_id]["normalized_cer"]
            row[f"{short}_hypothesis"] = by_sample[sample_id]["hypothesis"]
        candidates.append(row)

    for sample_id in large:
        cers = {model: rows[sample_id]["normalized_cer"] for model, rows in model_rows.items()}
        if cers["faster-whisper-large-v3"] == 0 and all(
            value > 0 for model, value in cers.items() if model != "faster-whisper-large-v3"
        ):
            add("large_v3_only_exact", sample_id, min(value for model, value in cers.items() if model != "faster-whisper-large-v3"))
        regression = cers["faster-whisper-large-v3-turbo"] - cers["faster-whisper-large-v3"]
        if regression >= 0.10:
            add("turbo_regression_ge_10pt", sample_id, regression)
        if cers["kotoba-whisper-v2.2"] < min(
            value for model, value in cers.items() if model != "kotoba-whisper-v2.2"
        ):
            next_best = min(
                value for model, value in cers.items() if model != "kotoba-whisper-v2.2"
            )
            add("kotoba_strictly_best", sample_id, next_best - cers["kotoba-whisper-v2.2"])
        if cers["nemotron-3.5-asr-streaming-0.6b"] < min(
            value for model, value in cers.items() if model != "nemotron-3.5-asr-streaming-0.6b"
        ):
            next_best = min(
                value
                for model, value in cers.items()
                if model != "nemotron-3.5-asr-streaming-0.6b"
            )
            add(
                "nemotron_streaming_strictly_best",
                sample_id,
                next_best - cers["nemotron-3.5-asr-streaming-0.6b"],
            )
        if cers["nemotron-3.5-asr-offline-transformers"] < min(
            value
            for model, value in cers.items()
            if model != "nemotron-3.5-asr-offline-transformers"
        ):
            next_best = min(
                value
                for model, value in cers.items()
                if model != "nemotron-3.5-asr-offline-transformers"
            )
            add(
                "nemotron_offline_strictly_best",
                sample_id,
                next_best - cers["nemotron-3.5-asr-offline-transformers"],
            )
        for model_id, category in (
            ("parakeet-tdt-ctc-0.6b-ja", "parakeet_strictly_best"),
            ("reazonspeech-k2-v2", "reazonspeech_strictly_best"),
        ):
            if cers[model_id] < min(
                value for other_model, value in cers.items() if other_model != model_id
            ):
                next_best = min(
                    value
                    for other_model, value in cers.items()
                    if other_model != model_id
                )
                add(category, sample_id, next_best - cers[model_id])
        offline_improvement = (
            cers["nemotron-3.5-asr-streaming-0.6b"]
            - cers["nemotron-3.5-asr-offline-transformers"]
        )
        if offline_improvement >= 0.10:
            add("nemotron_offline_improvement_ge_10pt", sample_id, offline_improvement)
        streaming_improvement = -offline_improvement
        if streaming_improvement >= 0.10:
            add("nemotron_streaming_improvement_ge_10pt", sample_id, streaming_improvement)
        for model_id, rows in model_rows.items():
            detail = rows[sample_id]
            if not detail["hypothesis"].strip():
                add(f"empty_output:{model_id}", sample_id, 1.0)
            if has_suspicious_repetition(detail["hypothesis"]):
                add(f"suspicious_repetition:{model_id}", sample_id, detail["normalized_cer"])
            reference_characters = detail["normalized_edits"]["reference_characters"]
            insertion_ratio = detail["normalized_edits"]["insertions"] / max(
                1, reference_characters
            )
            if insertion_ratio >= 0.20:
                add(f"high_insertion_ratio:{model_id}", sample_id, insertion_ratio)

    frame = pd.DataFrame(candidates)
    if not frame.empty:
        frame = frame.sort_values(["category", "score"], ascending=[True, False])
        frame = frame.groupby("category", as_index=False).head(20)
    frame.to_csv(output_dir / "contrast-candidates.csv", index=False)


def write_markdown_report(
    by_batch: pd.DataFrame, summary: pd.DataFrame, output_path: Path
) -> None:
    def value(row: pd.Series, key: str, digits: int = 4) -> str:
        item = row[key]
        return "-" if pd.isna(item) else f"{float(item):.{digits}f}"

    lines = [
        "# Aggregated benchmark results",
        "",
        "## Model summary",
        "",
        "| Model | Scope | Accuracy runs / BS | Raw CER | Normalized CER | Exact match | Performance runs / BS | Best BS | Audio hours/hour | RTF | Peak VRAM (MiB) | Pareto |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | :---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| {model} | {scope} | {accuracy_coverage} | {raw} | {normalized} | {exact} | {performance_coverage} | {bs} | {throughput} | {rtf} | {vram} | {pareto} |".format(
                model=row["display_name"],
                scope=row["comparison_scope"],
                accuracy_coverage="{runs} / {batch_sizes}".format(
                    runs=int(row["accuracy_runs"]),
                    batch_sizes=", ".join(
                        str(item) for item in row["accuracy_batch_sizes"]
                    ),
                ),
                raw=value(row, "raw_cer"),
                normalized=value(row, "normalized_cer"),
                exact=value(row, "exact_match_rate"),
                performance_coverage="{runs} / {batch_sizes}".format(
                    runs=int(row["performance_runs"]),
                    batch_sizes=", ".join(
                        str(item) for item in row["performance_batch_sizes"]
                    ),
                ),
                bs=int(row["best_batch_size"]),
                throughput=value(row, "audio_hours_per_hour", 2),
                rtf=value(row, "rtf", 6),
                vram=int(row["peak_vram_mib_nvidia_smi"]),
                pareto="yes" if row["pareto_frontier"] else "no",
            )
        )

    lines.extend(
        [
            "",
            "## Batch scaling",
            "",
            "| Model | BS | Runs | Audio hours/hour | RTF | Files/sec | Peak VRAM (MiB) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in by_batch.iterrows():
        lines.append(
            "| {model} | {bs} | {runs} | {throughput} | {rtf} | {files} | {vram} |".format(
                model=row["model_id"],
                bs=int(row["batch_size"]),
                runs=int(row["runs"]),
                throughput=value(row, "audio_hours_per_hour", 2),
                rtf=value(row, "rtf", 6),
                files=value(row, "files_per_sec", 2),
                vram=int(row["peak_vram_mib_nvidia_smi"]),
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_accuracy_markdown_report(summary: pd.DataFrame, output_path: Path) -> None:
    def value(row: pd.Series, key: str, digits: int = 4) -> str:
        item = row[key]
        return "-" if pd.isna(item) else f"{float(item):.{digits}f}"

    lines = [
        "# Accuracy results",
        "",
        "| Model | Scope | Runs | BS | Raw CER mean | Normalized CER mean | Normalized CER range | Std | Exact match | Empty output |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| {model} | {scope} | {runs} | {batch_sizes} | {raw} | {normalized} | {normalized_range} | {std} | {exact} | {empty} |".format(
                model=row["display_name"],
                scope=row["comparison_scope"],
                runs=int(row["runs"]),
                batch_sizes=", ".join(str(value) for value in row["batch_sizes"]),
                raw=value(row, "raw_cer"),
                normalized=value(row, "normalized_cer"),
                normalized_range="{minimum}–{maximum}".format(
                    minimum=value(row, "normalized_cer_min"),
                    maximum=value(row, "normalized_cer_max"),
                ),
                std=value(row, "normalized_cer_std"),
                exact=value(row, "exact_match_rate"),
                empty=value(row, "empty_output_rate"),
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accuracy-input-dir", type=Path)
    parser.add_argument("--performance-input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.accuracy_input_dir is None and args.performance_input_dir is None:
        parser.error("Specify at least one of --accuracy-input-dir or --performance-input-dir")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    accuracy_summary: pd.DataFrame | None = None
    performance_by_batch: pd.DataFrame | None = None
    performance_summary: pd.DataFrame | None = None

    if args.accuracy_input_dir is not None:
        accuracy_metadata = load_metadata(
            args.accuracy_input_dir,
            include_accuracy=True,
        )
        accuracy_summary = make_accuracy_summary(accuracy_metadata)
        pd.json_normalize(accuracy_metadata).to_csv(
            args.output_dir / "accuracy-runs.csv", index=False
        )
        accuracy_summary.to_csv(args.output_dir / "accuracy-summary.csv", index=False)
        write_json(
            args.output_dir / "accuracy-summary.json",
            accuracy_summary.to_dict(orient="records"),
        )
        extract_worst_cases(accuracy_metadata, args.output_dir)
        extract_contrasts(accuracy_metadata, args.output_dir)
        write_accuracy_markdown_report(
            accuracy_summary,
            args.output_dir / "accuracy-summary.md",
        )
        save_accuracy_figure(accuracy_summary, args.figure_dir)

    if args.performance_input_dir is not None:
        performance_metadata = load_metadata(
            args.performance_input_dir,
            include_accuracy=False,
        )
        performance_by_batch = aggregate_by_batch(performance_metadata)
        performance_summary = make_performance_summary(performance_by_batch)
        pd.json_normalize(performance_metadata).to_csv(
            args.output_dir / "performance-runs.csv", index=False
        )
        performance_by_batch.to_csv(args.output_dir / "by-batch.csv", index=False)
        performance_summary.to_csv(
            args.output_dir / "performance-summary.csv", index=False
        )
        write_json(
            args.output_dir / "performance-summary.json",
            performance_summary.to_dict(orient="records"),
        )
        save_figures(performance_by_batch, performance_summary, args.figure_dir)

    if accuracy_summary is not None and performance_summary is not None:
        summary = make_model_summary(accuracy_summary, performance_summary)
        summary.to_csv(args.output_dir / "model-summary.csv", index=False)
        write_json(args.output_dir / "model-summary.json", summary.to_dict(orient="records"))
        write_markdown_report(performance_by_batch, summary, args.output_dir / "summary.md")
        save_figures(performance_by_batch, summary, args.figure_dir)


if __name__ == "__main__":
    main()
