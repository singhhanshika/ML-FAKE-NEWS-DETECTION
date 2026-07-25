"""Reusable text normalization for training and inference."""

from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Iterable
from typing import Any

from sklearn.base import BaseEstimator, TransformerMixin

URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", re.IGNORECASE)
HTML_RE = re.compile(r"<[^>]+>")
SOURCE_PREFIX_RE = re.compile(
    r"^\s*(?:[A-Z][A-Z .'-]{1,35})?\s*\((?:Reuters|AP)\)\s*[-–—:]\s*",
    re.IGNORECASE,
)
SOURCE_MARKER_RE = re.compile(
    r"\b(?:reuters|associated\s+press|copyright\s+\d{4})\b",
    re.IGNORECASE,
)
REPEATED_CHAR_RE = re.compile(r"([a-z])\1{3,}", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")
NON_TEXT_RE = re.compile(r"[^\w\s!?'\-<>]", re.UNICODE)

CONTRACTIONS = {
    "can't": "can not",
    "cannot": "can not",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'ve": " have",
    "'ll": " will",
    "'d": " would",
    "'m": " am",
    "it's": "it is",
}


def normalize_text(
    text: object,
    *,
    lowercase: bool = True,
    replace_urls: bool = True,
    replace_numbers: bool = False,
    remove_source_markers: bool = True,
) -> str:
    """Normalize one document without discarding meaningful negation."""
    if text is None:
        return ""
    value = unicodedata.normalize("NFKC", str(text))
    value = value.replace("’", "'").replace("‘", "'")
    value = html.unescape(value)
    value = HTML_RE.sub(" ", value)
    value = EMAIL_RE.sub(" emailtoken ", value)
    value = URL_RE.sub(" urltoken " if replace_urls else " ", value)
    if remove_source_markers:
        value = SOURCE_PREFIX_RE.sub(" ", value)
        value = SOURCE_MARKER_RE.sub(" ", value)
    if lowercase:
        value = value.lower()
    for contraction, expansion in CONTRACTIONS.items():
        value = value.replace(contraction, expansion)
    value = REPEATED_CHAR_RE.sub(r"\1\1\1", value)
    if replace_numbers:
        value = re.sub(r"\b\d+(?:[.,]\d+)*\b", " numbertoken ", value)
    value = NON_TEXT_RE.sub(" ", value)
    value = value.replace("_", " ")
    return WHITESPACE_RE.sub(" ", value).strip()


class CleanTextTransformer(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer that applies the shared normalizer."""

    def __init__(
        self,
        lowercase: bool = True,
        replace_urls: bool = True,
        replace_numbers: bool = False,
        remove_source_markers: bool = True,
    ) -> None:
        self.lowercase = lowercase
        self.replace_urls = replace_urls
        self.replace_numbers = replace_numbers
        self.remove_source_markers = remove_source_markers

    def fit(self, X: Iterable[Any], y: object = None) -> CleanTextTransformer:
        """Return the stateless fitted transformer."""
        return self

    def transform(self, X: Iterable[Any]) -> list[str]:
        """Normalize every input document."""
        return [
            normalize_text(
                value,
                lowercase=self.lowercase,
                replace_urls=self.replace_urls,
                replace_numbers=self.replace_numbers,
                remove_source_markers=self.remove_source_markers,
            )
            for value in X
        ]
