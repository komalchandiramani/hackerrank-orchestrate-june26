# Multi-Modal Evidence Review

Verifies damage claims (car, laptop, package) by checking submitted images against a
claim conversation, with grounding from an evidence checklist and user history.

## Problem

Each claim has one or more images, a short support-chat transcript, and an object type.
For every claim in `dataset/claims.csv` the system produces one `output.csv` row deciding
whether the images **support**, **contradict**, or give **not enough information** for the
claim — plus the visible issue, the object part, evidence sufficiency, risk flags, severity,
and a short grounded justification. Images are the source of truth; the conversation only
says what to check; user history adds risk context but never overrides the visual evidence.

## Solution architecture

One vision-LLM (VLM) call per claim, wrapped in deterministic layers:

1. **Input build** — images are decoded (formats are mixed: AVIF/TIFF/PNG/WebP/JPEG behind
   `.jpg` names), downscaled, and re-encoded to JPEG. Each image is labelled with its image
   id. The matching rows of `evidence_requirements.csv` are injected so the model judges
   sufficiency against the real standard. User history is **not** sent to the model.
2. **Model call** — `claude-opus-4-6` returns a `ClaimDecision` via structured output
   (`messages.parse`), so the response is schema-valid by construction. `temperature=0`.
3. **No-image gate** — if no image is readable, skip the call and return a safe default.
4. **Deterministic post-processing**
   - `manual_review_required` is added from the user's `history_flags`.
   - A high-precision multilingual regex backstops `text_instruction_present` on explicit
     override phrasing in the chat (e.g. "ignore previous instructions", "approve the
     claim").

**Prompt-injection defence** is two-layered: the system prompt tells the model to treat all
chat/image text as untrusted data (never instructions) and to flag steering text without
obeying it; the regex is a deterministic backstop for chat overrides. Image-text injections
rely on the model.

## Codebase

```
code/
├── main.py        # entry point: picks sample vs test set, runs the pipeline
├── config.py      # model ids, pricing, USD->SGD, TEST_MODE flag
├── schema.py      # ClaimDecision (Pydantic) + CSV serialization
├── prompts.py     # system prompt, judge prompt, image encoding, message builder
├── pipeline.py    # per-claim processing, deterministic layers, run loop, report
├── metrics.py     # Stats accumulator + operational report builder
├── evaluation/
│   ├── main.py              # scores predictions vs labels -> results.json
│   └── evaluation_report.md # methodology, model comparison, operational analysis
└── requirements.txt
```

`schema.py` is the contract: categorical fields use `Literal`, multi-value fields
(`risk_flags`, `supporting_image_ids`) are lists serialized to the `;`-joined / `none` CSV
form, and `to_csv_row()` emits the 14 columns in order.

## Run

```bash
pip install -r code/requirements.txt
echo "ANTHROPIC_API_KEY=sk-ant-..." > code/.env

python code/main.py        # TEST_MODE=True -> sample_claims.csv -> sample_output.csv + sample_report.json
                           # TEST_MODE=False -> claims.csv -> output.csv + report.json
python code/evaluation/main.py   # scores sample_output.csv -> results.json
```

`TEST_MODE` lives in `config.py`. Each run also writes an operational report
(`*_report.json`): model calls, images sent, token usage, cost in SGD, and runtime.

## Evaluation strategy

`evaluation/main.py` scores predictions against the labelled `sample_claims.csv`, matched on
`(user_id, image_paths)`, with a metric chosen per field type:

- **Exact match** — booleans and single-value categoricals (`claim_status`, `issue_type`,
  `object_part`, `severity`, `evidence_standard_met`, `valid_image`).
- **IoU** — multi-value fields (`risk_flags`, `supporting_image_ids`).
- **LLM-as-judge** — the two free-text explanations, scored 0–100 by a **fixed** judge
  (`claude-sonnet-4-6`, `temperature=0`) so scores stay comparable across generation models.

Results (per-claim + aggregate means) are written to `results.json`. Two generation
configurations were compared (Sonnet 4.6 baseline vs Opus 4.6 + grounding + deterministic
layers); see `evaluation/evaluation_report.md` for the comparison and operational analysis.
