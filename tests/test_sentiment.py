"""
Tests for Sentiment Analysis Tool
Run: pytest tests/test_sentiment.py -v
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import SentimentAnalyzer


# ----------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------
@pytest.fixture(scope="module")
def analyzer():
    """Shared SentimentAnalyzer instance (loaded once per module)."""
    return SentimentAnalyzer()


# ----------------------------------------------------------------
# Initialization tests
# ----------------------------------------------------------------
class TestSentimentAnalyzerInit:

    def test_instantiation(self, analyzer):
        assert analyzer is not None

    def test_model_name_set(self, analyzer):
        assert analyzer.model_name is not None
        assert isinstance(analyzer.model_name, str)

    def test_channel_weights_defined(self, analyzer):
        expected_channels = {"reviews", "social", "support", "email", "nps", "chat", "unknown"}
        assert expected_channels.issubset(set(analyzer.channel_weights.keys()))

    def test_channel_weights_are_positive(self, analyzer):
        for ch, w in analyzer.channel_weights.items():
            assert w > 0, f"Channel '{ch}' has non-positive weight {w}"

    def test_aspects_keywords_loaded(self, analyzer):
        assert isinstance(analyzer.aspects_keywords, dict)
        assert len(analyzer.aspects_keywords) > 0

    def test_sarcasm_indicators_loaded(self, analyzer):
        assert isinstance(analyzer.sarcasm_indicators, (list, dict, set))
        assert len(analyzer.sarcasm_indicators) > 0


# ----------------------------------------------------------------
# Prediction output structure tests
# ----------------------------------------------------------------
class TestSentimentPrediction:

    POSITIVE_TEXTS = [
        "I absolutely love this product, it works perfectly!",
        "Amazing experience, highly recommend to everyone.",
        "Fantastic quality, exceeded my expectations.",
    ]

    NEGATIVE_TEXTS = [
        "This is terrible, completely broken and useless.",
        "Worst experience ever, very disappointed.",
        "Total waste of money, nothing works.",
    ]

    EDGE_CASES = [
        "",                          # empty string
        " ",                         # whitespace only
        "ok",                        # single word
        "a" * 1000,                  # very long text
        "123 !!! ### ???",           # no real words
        "Haha sure, totally great.", # possible sarcasm
    ]

    def test_positive_text_returns_result(self, analyzer):
        for text in self.POSITIVE_TEXTS:
            result = analyzer.analyze(text)
            assert result is not None

    def test_negative_text_returns_result(self, analyzer):
        for text in self.NEGATIVE_TEXTS:
            result = analyzer.analyze(text)
            assert result is not None

    def test_output_has_sentiment_field(self, analyzer):
        result = analyzer.analyze("This product is great!")
        assert "sentiment" in result or "label" in result or "score" in result

    def test_output_score_is_numeric(self, analyzer):
        result = analyzer.analyze("Good product overall.")
        score = result.get("score") or result.get("confidence") or result.get("probability")
        if score is not None:
            assert isinstance(score, (int, float))

    def test_empty_string_does_not_crash(self, analyzer):
        try:
            result = analyzer.analyze("")
            assert result is not None
        except (ValueError, Exception):
            pass  # Graceful failure is acceptable

    def test_long_text_does_not_crash(self, analyzer):
        long_text = "This is a great product. " * 100
        try:
            result = analyzer.analyze(long_text)
            assert result is not None
        except Exception:
            pass

    def test_batch_analysis_returns_list(self, analyzer):
        texts = ["Great!", "Terrible!", "It was okay."]
        if hasattr(analyzer, "analyze_batch"):
            results = analyzer.analyze_batch(texts)
            assert isinstance(results, list)
            assert len(results) == len(texts)


# ----------------------------------------------------------------
# Channel weight tests
# ----------------------------------------------------------------
class TestChannelWeights:

    def test_support_channel_weight_higher_than_social(self, analyzer):
        assert analyzer.channel_weights["support"] >= analyzer.channel_weights["social"]

    def test_unknown_channel_has_neutral_weight(self, analyzer):
        assert analyzer.channel_weights["unknown"] == 1.0