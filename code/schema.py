"""Structured output schema for the multi-modal evidence review system."""

from __future__ import annotations

from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator

# --------------------------------------------------------------------------- #
# Categorical value sets (problem_statement.md lines 130-140)
# --------------------------------------------------------------------------- #

# Input object type
ClaimObject = Literal["car", "laptop", "package"]

# Final decision
ClaimStatus = Literal["supported", "contradicted", "not_enough_information"]

# Visible issue type
IssueType = Literal[
    "dent",
    "scratch",
    "crack",
    "glass_shatter",
    "broken_part",
    "missing_part",
    "torn_packaging",
    "crushed_packaging",
    "water_damage",
    "stain",
    "none",
    "unknown",
]

# Object parts are object-specific. The output is a single
# `object_part` column, so the field accepts the union of all object parts.
CarPart = Literal[
    "front_bumper",
    "rear_bumper",
    "door",
    "hood",
    "windshield",
    "side_mirror",
    "headlight",
    "taillight",
    "fender",
    "quarter_panel",
    "body",
    "unknown",
]
LaptopPart = Literal[
    "screen",
    "keyboard",
    "trackpad",
    "hinge",
    "lid",
    "corner",
    "port",
    "base",
    "body",
    "unknown",
]
PackagePart = Literal[
    "box",
    "package_corner",
    "package_side",
    "seal",
    "label",
    "contents",
    "item",
    "unknown",
]
ObjectPart = CarPart | LaptopPart | PackagePart

# Risk flags. `none` is the sentinel for "no flags".
RiskFlag = Literal[
    "none",
    "blurry_image",
    "cropped_or_obstructed",
    "low_light_or_glare",
    "wrong_angle",
    "wrong_object",
    "wrong_object_part",
    "damage_not_visible",
    "claim_mismatch",
    "possible_manipulation",
    "non_original_image",
    "text_instruction_present",
    "user_history_risk",
    "manual_review_required",
]

# Severity
Severity = Literal["none", "low", "medium", "high", "unknown"]

# Maps the input `claim_object` to its set of valid parts, for cross-field
# validation and for constraining a VLM to object-appropriate parts.
PARTS_BY_OBJECT: dict[str, tuple[str, ...]] = {
    "car": get_args(CarPart),
    "laptop": get_args(LaptopPart),
    "package": get_args(PackagePart),
}

# The exact output column order
CSV_COLUMNS: tuple[str, ...] = (
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


class ClaimDecision(BaseModel):
    # The 10 predicted fields of an output.csv row; inputs are supplied at to_csv_row().
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # --- evidence sufficiency --------------------------------------------- #
    evidence_standard_met: bool = Field(
        description="True if the image set is sufficient to evaluate the claim."
    )
    evidence_standard_met_reason: str = Field(
        description="Short reason for the evidence-sufficiency decision."
    )

    # --- risk -------------------------------------------------------------- #
    risk_flags: list[RiskFlag] = Field(
        default_factory=lambda: ["none"],
        description="Risk flags; serializes to a semicolon-separated string or 'none'.",
    )

    # --- visual findings --------------------------------------------------- #
    issue_type: IssueType
    object_part: ObjectPart

    # --- decision ---------------------------------------------------------- #
    claim_status: ClaimStatus
    claim_status_justification: str = Field(
        description="Concise, image-grounded explanation; may reference image IDs."
    )
    supporting_image_ids: list[str] = Field(
        default_factory=list,
        description="Image IDs (filename without extension) supporting the decision.",
    )

    valid_image: bool = Field(
        description="True if the image set is usable for automated review."
    )
    severity: Severity = "unknown"

    @field_validator("risk_flags", mode="before")
    @classmethod
    def _coerce_risk_flags(cls, v: object) -> list[str]:
        # Accept "none" / "a;b;c" / list; normalize to a list.
        if v is None:
            return ["none"]
        if isinstance(v, str):
            v = [part.strip() for part in v.split(";") if part.strip()]
        return list(v) if v else ["none"]

    @field_validator("risk_flags")
    @classmethod
    def _clean_risk_flags(cls, v: list[str]) -> list[str]:
        # Drop the "none" sentinel when real flags exist; dedupe, keep order.
        real = [f for f in v if f != "none"]
        if not real:
            return ["none"]
        seen: set[str] = set()
        deduped = [f for f in real if not (f in seen or seen.add(f))]
        return deduped

    @field_validator("supporting_image_ids", mode="before")
    @classmethod
    def _coerce_supporting_ids(cls, v: object) -> list[str]:
        # Accept "none" / "img_1;img_2" / list; normalize.
        if v is None:
            return []
        if isinstance(v, str):
            if v.strip().lower() == "none":
                return []
            return [part.strip() for part in v.split(";") if part.strip()]
        return list(v)

    @staticmethod
    def _bool_str(value: bool) -> str:
        return "true" if value else "false"

    def to_csv_row(
        self,
        user_id: str,
        image_paths: str,
        user_claim: str,
        claim_object: ClaimObject,
    ) -> dict[str, str]:
        # Builds the full 14-column row from the inputs + predicted fields.
        # Ordered to match CSV_COLUMNS; lowercase bools, ";"-joined lists, "none" sentinels.
        # Coerce a part that isn't valid for the claimed object to "unknown".
        object_part = (
            self.object_part
            if self.object_part in PARTS_BY_OBJECT[claim_object]
            else "unknown"
        )
        return {
            "user_id": user_id,
            "image_paths": image_paths,
            "user_claim": user_claim,
            "claim_object": claim_object,
            "evidence_standard_met": self._bool_str(self.evidence_standard_met),
            "evidence_standard_met_reason": self.evidence_standard_met_reason,
            "risk_flags": ";".join(self.risk_flags) if self.risk_flags else "none",
            "issue_type": self.issue_type,
            "object_part": object_part,
            "claim_status": self.claim_status,
            "claim_status_justification": self.claim_status_justification,
            "supporting_image_ids": ";".join(self.supporting_image_ids)
            if self.supporting_image_ids
            else "none",
            "valid_image": self._bool_str(self.valid_image),
            "severity": self.severity,
        }
