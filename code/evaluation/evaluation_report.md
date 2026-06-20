# Evaluation Report — Multi-Modal Evidence Review

## 1. System summary

For each claim the pipeline sends the submitted image(s) (each preceded by its image id),
the claim conversation, the claim object, and a per-object minimum-evidence checklist to a
single vision LLM call, which returns a validated `ClaimDecision` (structured output). Two
deterministic layers run around the model:

- **Evidence grounding** — the matching rows of `evidence_requirements.csv` are injected so
  the model judges `evidence_standard_met` against the actual standard.
- **Post-processing** — `manual_review_required` is added from `user_history.csv`; a
  high-precision multilingual regex backstops `text_instruction_present` on explicit
  override phrasing in chat. A no-readable-images gate returns a safe default without a
  model call.

## 2. Evaluation methodology

`code/evaluation/main.py` scores predictions (`sample_output.csv`) against the labels in
`dataset/sample_claims.csv`, matched by `(user_id, image_paths)`:

- **Exact match** (1/0): `evidence_standard_met`, `valid_image`, `issue_type`,
  `object_part`, `claim_status`, `severity`.
- **IoU** (multi-value): `risk_flags` (`none` is a token), `supporting_image_ids`
  (`none` = empty set; two empties = 1.0).
- **LLM-as-judge** (0–100): `evidence_standard_met_reason`, `claim_status_justification`.
  One call per claim per metric, **fixed judge = `claude-sonnet-4-6`, temperature 0**, kept
  constant across generation models so scores stay comparable. The judge sees only the
  reference and predicted explanation.

## 3. Strategies compared (sample set, n=20)

Two configurations were evaluated end-to-end on the labeled sample:

- **Config A — baseline:** `claude-sonnet-4-6`, prompt-only (no evidence grounding, no
  deterministic layers).
- **Config B — final:** `claude-opus-4-6`, temperature 0, with evidence-requirement
  grounding and the history/injection post-processing layers.

| Metric | A: Sonnet 4.6 | B: Opus 4.6 (final) |
|---|---|---|
| evidence_standard_met (acc) | 0.85 | 0.85 |
| valid_image (acc) | 0.95 | 0.95 |
| issue_type (acc) | 0.65 | 0.55 |
| object_part (acc) | 0.85 | 0.75 |
| claim_status (acc) | 0.75 | 0.75 |
| severity (acc) | 0.25 | 0.45 |
| risk_flags (IoU) | 0.29 | 0.59 |
| supporting_image_ids (IoU) | 0.83 | 0.83 |
| evidence_standard_met_reason (judge) | 71.7 | 75.1 |
| claim_status_justification (judge) | 65.1 | 72.2 |

**Chosen: Config B.** It materially improves the hardest fields — `risk_flags` IoU roughly
doubles (largely the deterministic history/injection layers), `severity` improves, and both
judge scores rise — at the cost of a small dip on `issue_type`/`object_part`. The risk and
justification gains matter most for this task (correct routing + grounded explanations).

**Known weak spots:** `severity` is the lowest exact-match field (subjective boundary
between low/medium/high); `issue_type` regressed slightly under Opus and is worth a targeted
prompt pass; `risk_flags` IoU remains the main headroom.

## 4. Operational analysis

Measured on the sample set with Config B (`sample_report.json`); pricing assumptions are
Opus 4.6 ($5 in / $25 out per MTok) converted at USD→SGD 1.35.

| | Sample (n=20, measured) | Full test set (n=44, measured) |
|---|---|---|
| Model calls | 20 (1.0 / claim) | 44 (1.0 / claim) |
| Images sent | 29 | 82 |
| Input tokens | 84,138 | 197,411 |
| Output tokens | 4,250 | 9,943 |
| Cost (SGD) | 0.7114 | 1.6681 |
| Avg cost / claim (SGD) | 0.0356 | 0.0379 |
| Runtime | 155.8 s | 379.3 s (~6.3 min) |
| Avg latency / claim | 7.8 s | 8.6 s |

Both runs use Config B (Opus 4.6). Cost scales linearly at ~0.037–0.038 SGD/claim; the full
44-claim test set processes for **~1.67 SGD in ~6.3 min** with no caching.

**Calls / cost / latency:** exactly one model call per claim (the judge adds 2 calls/claim
but only during evaluation, not production). No caching is used — the system prompt is
identical across calls, so a `cache_control` breakpoint on it would cut input-token cost
substantially on larger runs.

**TPM/RPM / scaling:** the run is sequential (one claim at a time), so rate limits are not a
concern at this volume; per-claim image payloads dominate input tokens. For larger batches:
prompt-cache the static system prompt + requirements block, run with bounded concurrency,
and rely on the SDK's built-in 429/5xx retry with backoff. The Batches API (50% cost) is an
option for non-latency-sensitive full-test runs.

## 5. Reproducibility

```bash
pip install -r code/requirements.txt   # anthropic, pydantic, python-dotenv, tqdm
echo "ANTHROPIC_API_KEY=sk-ant-..." > code/.env
python code/main.py                    # TEST_MODE in config.py: sample vs claims.csv
python code/evaluation/main.py         # -> results.json
```
