# src/parsing/__init__.py

# Purpose: Expose parsing-related classes and functions for external use.

# Input: This module imports and re-exports the main parsing components:
# - LogParser (interface)
# - RegexLogParser (regex-based implementation)
# - JsonLogParser (JSON-based implementation)
# - TemplateMiner (for mining templates from log messages)
# - EventTokenizer (for encoding template IDs into token IDs)

# Output: Users can import these classes directly from src.parsing, e.g.:
# from src.parsing import TemplateMiner, EventTokenizer

# Used by: Other stages of the pipeline that need to parse raw logs, mine templates, or tokenize events.

from .parsers import LogParser, RegexLogParser, JsonLogParser
from .template_miner import TemplateMiner
from .tokenizer import EventTokenizer

__all__ = [
    "LogParser", "RegexLogParser", "JsonLogParser",
    "TemplateMiner", "EventTokenizer",
]
