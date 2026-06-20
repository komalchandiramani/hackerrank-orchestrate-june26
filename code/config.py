# Test mode (default) runs on the labeled sample set; set False for the real test set.
TEST_MODE = False

# Claude model + request settings (API key is read from ANTHROPIC_API_KEY env var).
MODEL = "claude-opus-4-6"
MAX_TOKENS = 2048

# Fixed judge model, kept constant across generation-model experiments for fair A/B.
JUDGE_MODEL = "claude-sonnet-4-6"

# Generation-model pricing (USD per 1M tokens) and the USD->SGD rate for cost reporting.
# Opus 4.6: $5 in / $25 out.
INPUT_USD_PER_MTOK = 5.0
OUTPUT_USD_PER_MTOK = 25.0
CACHE_READ_USD_PER_MTOK = 0.50
CACHE_WRITE_USD_PER_MTOK = 6.25
USD_TO_SGD = 1.35
