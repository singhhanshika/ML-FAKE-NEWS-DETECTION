"""TF-IDF feature construction."""

from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline

from fake_news_detector.preprocessing import CleanTextTransformer


def build_feature_pipeline(config: dict[str, Any]) -> Pipeline:
    """Build word and optional character TF-IDF features."""
    vectorizer = config["vectorizer"]
    preprocessing = config["preprocessing"]
    word = vectorizer["word"]
    preserved_negations = set(preprocessing.get("preserve_negations", []))
    stop_words = sorted(set(ENGLISH_STOP_WORDS) - preserved_negations)
    transformers: list[tuple[str, TfidfVectorizer]] = [
        (
            "word",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=tuple(word["ngram_range"]),
                min_df=word["min_df"],
                max_df=word["max_df"],
                max_features=word["max_features"],
                sublinear_tf=word["sublinear_tf"],
                strip_accents=word["strip_accents"],
                norm=word["norm"],
                stop_words=stop_words,
                token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z'-]+\b",
            ),
        )
    ]
    character = vectorizer["character"]
    if character.get("enabled", False):
        transformers.append(
            (
                "character",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=tuple(character["ngram_range"]),
                    min_df=character["min_df"],
                    max_features=character["max_features"],
                    sublinear_tf=character["sublinear_tf"],
                    norm="l2",
                ),
            )
        )
    cleaner = CleanTextTransformer(
        lowercase=preprocessing["lowercase"],
        replace_urls=preprocessing["replace_urls"],
        replace_numbers=preprocessing["replace_numbers"],
        remove_source_markers=preprocessing["remove_source_markers"],
    )
    return Pipeline(
        [
            ("clean", cleaner),
            ("tfidf", FeatureUnion(transformers)),
        ]
    )
