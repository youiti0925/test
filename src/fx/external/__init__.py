"""External comparator subpackage (spec §9).

`ExternalSignal` and `ComparisonResult` live in `comparison.py`; readers
for specific vendors live in their own modules. Per spec §9, this code
NEVER scrapes — only ingests official APIs / CSV / Webhook payloads
the user has explicit permission to use.
"""
from .comparison import (
    ComparisonResult,
    ExternalSignal,
    aggregate_comparisons,
    compare_signals,
)
from .csv_reader import read_external_csv

__all__ = [
    "ComparisonResult",
    "ExternalSignal",
    "aggregate_comparisons",
    "compare_signals",
    "read_external_csv",
]
