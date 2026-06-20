from __future__ import annotations

from dataclasses import dataclass

from config import (
    CACHE_READ_USD_PER_MTOK,
    CACHE_WRITE_USD_PER_MTOK,
    INPUT_USD_PER_MTOK,
    MODEL,
    OUTPUT_USD_PER_MTOK,
    USD_TO_SGD,
)


@dataclass
class Stats:
    model_calls: int = 0
    images_sent: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def add_usage(self, usage) -> None:
        self.model_calls += 1
        self.input_tokens += usage.input_tokens or 0
        self.output_tokens += usage.output_tokens or 0
        self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0

    def cost_sgd(self) -> float:
        usd = (
            self.input_tokens * INPUT_USD_PER_MTOK
            + self.output_tokens * OUTPUT_USD_PER_MTOK
            + self.cache_read_tokens * CACHE_READ_USD_PER_MTOK
            + self.cache_write_tokens * CACHE_WRITE_USD_PER_MTOK
        ) / 1_000_000
        return usd * USD_TO_SGD


def build_report(stats: Stats, num_claims: int, runtime_s: float) -> dict:
    cost = stats.cost_sgd()
    return {
        "total_model_calls": stats.model_calls,
        "avg_model_calls_per_claim": round(stats.model_calls / num_claims, 4) if num_claims else 0,
        "total_images_sent": stats.images_sent,
        "total_cost_sgd": round(cost, 4),
        "avg_cost_sgd_per_claim": round(cost / num_claims, 6) if num_claims else 0,
        "runtime_seconds": round(runtime_s, 2),
        "total_claims": num_claims,
        "avg_runtime_seconds_per_claim": round(runtime_s / num_claims, 3) if num_claims else 0,
        "token_usage": {
            "input_tokens": stats.input_tokens,
            "output_tokens": stats.output_tokens,
            "cache_read_tokens": stats.cache_read_tokens,
            "cache_write_tokens": stats.cache_write_tokens,
        },
        "pricing_assumptions": {
            "model": MODEL,
            "input_usd_per_mtok": INPUT_USD_PER_MTOK,
            "output_usd_per_mtok": OUTPUT_USD_PER_MTOK,
            "cache_read_usd_per_mtok": CACHE_READ_USD_PER_MTOK,
            "cache_write_usd_per_mtok": CACHE_WRITE_USD_PER_MTOK,
            "usd_to_sgd": USD_TO_SGD,
        },
    }
