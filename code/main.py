from __future__ import annotations

from config import TEST_MODE
from pipeline import (
    SAMPLE_INPUT,
    SAMPLE_OUTPUT,
    SAMPLE_REPORT,
    TEST_INPUT,
    TEST_OUTPUT,
    TEST_REPORT,
    run,
)


def main() -> None:
    if TEST_MODE:
        run(SAMPLE_INPUT, SAMPLE_OUTPUT, SAMPLE_REPORT)
    else:
        run(TEST_INPUT, TEST_OUTPUT, TEST_REPORT)


if __name__ == "__main__":
    main()
