from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from tqdm import tqdm

CODE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from config import JUDGE_MODEL  # noqa: E402
from prompts import (  # noqa: E402
    JUDGE_CRITERIA_CLAIM_STATUS,
    JUDGE_CRITERIA_EVIDENCE,
    LLM_JUDGE_SYSTEM_PROMPT,
)

load_dotenv(CODE_DIR / ".env")

GROUND_TRUTH = REPO_ROOT / "dataset" / "sample_claims.csv"
PREDICTIONS = REPO_ROOT / "sample_output.csv"
RESULTS = REPO_ROOT / "results.json"

KEY_COLS = ("user_id", "image_paths")

# Booleans + single-value categoricals: exact match.
EQUALITY_COLS = (
    "evidence_standard_met",
    "valid_image",
    "issue_type",
    "object_part",
    "claim_status",
    "severity",
)
# Multi-value categoricals: intersection-over-union.
IOU_COLS = ("risk_flags", "supporting_image_ids")
# Free-text explanations: LLM-as-a-judge. metric -> criteria for the judge prompt.
JUDGE_COLS = {
    "evidence_standard_met_reason": JUDGE_CRITERIA_EVIDENCE,
    "claim_status_justification": JUDGE_CRITERIA_CLAIM_STATUS,
}

client = anthropic.Anthropic()


def _norm(value: str) -> str:
    return (value or "").strip().lower()


def _parse_set(value: str, none_is_empty: bool) -> set[str]:
    tokens = {t.strip().lower() for t in (value or "").split(";") if t.strip()}
    if none_is_empty:
        tokens.discard("none")
    return tokens


def iou(ref: str, pred: str, none_is_empty: bool) -> float:
    a = _parse_set(ref, none_is_empty)
    b = _parse_set(pred, none_is_empty)
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def judge(criteria: str, reference: str, predicted: str) -> int:
    system = LLM_JUDGE_SYSTEM_PROMPT.format(criteria=criteria)
    user = f"Reference explanation:\n{reference}\n\nPredicted explanation:\n{predicted}"
    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=16,
        temperature=0,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    match = re.search(r"\d+", text)
    return max(0, min(100, int(match.group()))) if match else 0


def _load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def main() -> None:
    truth = {tuple(r[k] for k in KEY_COLS): r for r in _load(GROUND_TRUTH)}
    predictions = _load(PREDICTIONS)

    per_claim = []
    for pred in tqdm(predictions, desc="Scoring claims", unit="claim"):
        ref = truth.get(tuple(pred[k] for k in KEY_COLS))
        if ref is None:
            continue

        equality = {c: int(_norm(ref[c]) == _norm(pred[c])) for c in EQUALITY_COLS}
        iou_scores = {
            c: round(iou(ref[c], pred[c], none_is_empty=(c == "supporting_image_ids")), 4)
            for c in IOU_COLS
        }
        judge_scores = {
            c: judge(criteria, ref[c], pred[c]) for c, criteria in JUDGE_COLS.items()
        }

        per_claim.append(
            {
                "user_id": pred["user_id"],
                "image_paths": pred["image_paths"],
                "equality": equality,
                "iou": iou_scores,
                "judge": judge_scores,
            }
        )

    aggregate = {
        "num_claims": len(per_claim),
        "equality": {c: _mean([r["equality"][c] for r in per_claim]) for c in EQUALITY_COLS},
        "iou": {c: _mean([r["iou"][c] for r in per_claim]) for c in IOU_COLS},
        "judge": {c: _mean([r["judge"][c] for r in per_claim]) for c in JUDGE_COLS},
    }

    RESULTS.write_text(json.dumps({"aggregate": aggregate, "per_claim": per_claim}, indent=2), encoding="utf-8")
    print(f"Scored {len(per_claim)} claims -> {RESULTS}")


if __name__ == "__main__":
    main()
