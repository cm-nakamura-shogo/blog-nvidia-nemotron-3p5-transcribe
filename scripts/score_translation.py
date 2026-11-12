#!/usr/bin/env python3
"""Score speech-translation probe output against FLEURS parallel transcripts.

FLEURS sentences share ``fleurs_id`` across languages, so the reference for a
Japanese recording is the English transcript with the same id (and vice versa).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import read_jsonl, write_json

TOKENIZERS = {"en": "13a", "ja": "ja-mecab"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--target-lang", choices=sorted(TOKENIZERS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import sacrebleu

    references: dict[int, str] = {}
    for row in read_jsonl(args.reference_manifest):
        # 同じ文を複数話者が読んでいるため、最初の書き起こしを参照訳とする。
        references.setdefault(int(row["fleurs_id"]), row["raw_transcription"])

    predictions = read_jsonl(args.predictions)
    pairs = [
        (row["pred_text"], references[int(row["fleurs_id"])])
        for row in predictions
        if int(row["fleurs_id"]) in references
    ]
    hypotheses = [hypothesis for hypothesis, _ in pairs]
    refs = [[reference for _, reference in pairs]]
    bleu_metric = sacrebleu.BLEU(tokenize=TOKENIZERS[args.target_lang])
    chrf_metric = sacrebleu.CHRF()
    bleu = bleu_metric.corpus_score(hypotheses, refs)
    chrf = chrf_metric.corpus_score(hypotheses, refs)
    write_json(
        args.output,
        {
            "predictions": str(args.predictions),
            "target_lang": args.target_lang,
            "files": len(predictions),
            "scored_files": len(pairs),
            "bleu": bleu.score,
            "bleu_signature": str(bleu_metric.get_signature()),
            "chrf": chrf.score,
        },
    )
    print(f"{args.predictions.name}: BLEU {bleu.score:.2f} chrF {chrf.score:.2f} ({len(pairs)}/{len(predictions)})")


if __name__ == "__main__":
    main()
