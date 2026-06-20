from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from tqdm import tqdm

from config import MAX_TOKENS, MODEL
from metrics import Stats, build_report
from prompts import SYSTEM_PROMPT, build_user_content
from schema import CSV_COLUMNS, ClaimDecision

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
DATASET_DIR = REPO_ROOT / "dataset"

# Load ANTHROPIC_API_KEY (and any other vars) from code/.env.
load_dotenv(CODE_DIR / ".env")

# Test mode runs on the labeled sample set; otherwise the real test set.
SAMPLE_INPUT = DATASET_DIR / "sample_claims.csv"
SAMPLE_OUTPUT = REPO_ROOT / "sample_output.csv"
SAMPLE_REPORT = REPO_ROOT / "sample_report.json"
TEST_INPUT = DATASET_DIR / "claims.csv"
TEST_OUTPUT = REPO_ROOT / "output.csv"
TEST_REPORT = REPO_ROOT / "report.json"

# history_flags values that trigger adding a deterministic risk flag.
HISTORY_FLAGS_REQUIRING_REVIEW = {"user_history_risk;manual_review_required"}


def _load_user_history() -> dict[str, str]:
    with (DATASET_DIR / "user_history.csv").open(newline="", encoding="utf-8") as f:
        return {r["user_id"]: r["history_flags"] for r in csv.DictReader(f)}


def _load_evidence_requirements() -> list[dict[str, str]]:
    with (DATASET_DIR / "evidence_requirements.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


USER_HISTORY = _load_user_history()
EVIDENCE_REQUIREMENTS = _load_evidence_requirements()


def build_requirements_block(claim_object: str) -> str:
    # Requirements whose claim_object matches this claim or applies to all objects.
    return "\n".join(
        f"- {r['applies_to']}: {r['minimum_image_evidence']}"
        for r in EVIDENCE_REQUIREMENTS
        if r["claim_object"] in (claim_object, "all")
    )

client = anthropic.Anthropic()


def apply_history_risk(decision: ClaimDecision, user_id: str) -> ClaimDecision:
    # Deterministic layer: flag manual review based on the user's claim history.
    if USER_HISTORY.get(user_id, "") in HISTORY_FLAGS_REQUIRING_REVIEW:
        decision.risk_flags = [*decision.risk_flags, "manual_review_required"]
    return decision


# High-precision backstop: explicit override imperatives in chat text only (not images).
# Matches override commands, not innocent words like "note". Sets the flag; never the verdict.
_INJECTION_RE = re.compile(
    "|".join(
        (
            r"ignore (all )?(the )?(previous|prior) instructions",
            r"approve (this|the|my) claim",
            r"mark (this|it|the row|the claim)?\s*(as\s+)?supported",
            r"mark as supported",
            # Hindi (romanized): "...claim approve kar dena/do"
            r"claim approve",
            r"approve kar (do|dena|dijiye)",
            # Spanish
            r"aprueb\w* (la |el )?(reclamaci[oó]n|reclamo)",
            r"marca\w* como (soportad|aprobad)\w*",
            r"ignora\w* (las )?instrucciones (anteriores|previas)",
            # Chinese
            r"批准(这|该)?(个)?(理赔|索赔|申请|claim)",
            r"标记为(支持|已支持)",
            r"忽略(之前|先前|以上)的?指[示令]",
        )
    ),
    re.IGNORECASE,
)


def apply_injection_flag(decision: ClaimDecision, user_claim: str) -> ClaimDecision:
    # Backstop over chat text only: force text_instruction_present on an override match.
    if _INJECTION_RE.search(user_claim or ""):
        decision.risk_flags = [*decision.risk_flags, "text_instruction_present"]
    return decision


def _safe_default(risk_flags: list[str] | None = None) -> ClaimDecision:
    # Fallback decision used when the claim cannot be evaluated (e.g. no readable images).
    return ClaimDecision(
        evidence_standard_met=False,
        evidence_standard_met_reason="No readable images submitted; the claim cannot be evaluated.",
        risk_flags=risk_flags or ["none"],
        issue_type="unknown",
        object_part="unknown",
        claim_status="not_enough_information",
        claim_status_justification="No readable images were submitted, so the claim cannot be verified.",
        supporting_image_ids=[],
        valid_image=False,
        severity="unknown",
    )


def process(row: dict[str, str], stats: Stats) -> ClaimDecision:
    content, num_images = build_user_content(
        image_paths=row["image_paths"],
        user_claim=row["user_claim"],
        claim_object=row["claim_object"],
        dataset_dir=DATASET_DIR,
        requirements_block=build_requirements_block(row["claim_object"]),
    )
    # No readable images: skip the model call and return a safe default.
    if num_images == 0:
        return _safe_default(risk_flags=["manual_review_required"])
    response = client.messages.parse(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
        output_format=ClaimDecision,
    )
    stats.images_sent += num_images
    stats.add_usage(response.usage)
    decision = apply_history_risk(response.parsed_output, row["user_id"])
    return apply_injection_flag(decision, row["user_claim"])


def run(input_path: Path, output_path: Path, report_path: Path) -> None:
    with input_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    stats = Stats()
    start = time.perf_counter()
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in tqdm(rows, desc="Processing claims", unit="claim"):
            decision = process(row, stats)
            writer.writerow(
                decision.to_csv_row(
                    user_id=row["user_id"],
                    image_paths=row["image_paths"],
                    user_claim=row["user_claim"],
                    claim_object=row["claim_object"],
                )
            )
    runtime_s = time.perf_counter() - start

    report = build_report(stats, len(rows), runtime_s)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote {len(rows)} rows to {output_path}")
    print(f"Wrote report to {report_path}")
