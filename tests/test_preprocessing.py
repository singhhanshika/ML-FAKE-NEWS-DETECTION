"""Tests for shared text normalization."""

from fake_news_detector.preprocessing import CleanTextTransformer, normalize_text


def test_normalizes_unicode_and_lowercase() -> None:
    assert normalize_text("ＣＡＦÉ News") == "café news"


def test_removes_html() -> None:
    assert normalize_text("<p>Important <strong>report</strong></p>") == "important report"


def test_replaces_url_and_removes_email() -> None:
    result = normalize_text("See https://example.com or editor@example.com")
    assert "https" not in result
    assert "example.com" not in result
    assert "urltoken" in result
    assert "emailtoken" in result


def test_preserves_negation() -> None:
    result = normalize_text("This isn't true and can’t be verified.")
    assert "not" in result


def test_transformer_handles_none() -> None:
    assert CleanTextTransformer().transform([None, " News "]) == ["", "news"]


def test_removes_known_source_marker() -> None:
    assert "reuters" not in normalize_text("LONDON (Reuters) - A report was issued.")
