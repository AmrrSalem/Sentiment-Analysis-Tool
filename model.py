import re
import numpy as np
import pandas as pd  # Make sure pandas is imported
import json
from typing import Dict, List, Any, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import train_test_split
    import torch

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

import emoji
import langdetect
from textblob import TextBlob
import hashlib
import pickle
import os


class SentimentAnalyzer:
    def __init__(self, model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment"):
        """
        Initialize sentiment analyzer with multilingual support and calibration

        Args:
            model_name: HuggingFace model name for sentiment analysis
        """
        self.model_name = model_name
        self.sentiment_pipeline = None
        self.calibrator = None
        self.aspects_keywords = self._load_aspect_keywords()
        self.sarcasm_indicators = self._load_sarcasm_indicators()
        self.channel_weights = {
            'reviews': 1.0,
            'social': 0.9,  # Often more casual/sarcastic
            'support': 1.1,  # Usually more direct
            'email': 1.0,
            'nps': 1.0,
            'chat': 0.95,  # More informal
            'unknown': 1.0
        }

        # Set random seed for reproducibility
        np.random.seed(42)
        if TRANSFORMERS_AVAILABLE:
            torch.manual_seed(42)

        self._initialize_model()

    def _initialize_model(self):
        """Initialize the sentiment analysis model"""
        if TRANSFORMERS_AVAILABLE:
            try:
                # Load multilingual sentiment model
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model=self.model_name,
                    tokenizer=self.model_name,
                    return_all_scores=True,
                    device=-1  # CPU only for broader compatibility
                )
                print("✅ Transformer model loaded successfully")
            except Exception as e:
                print(f"⚠️  Failed to load transformer model: {e}")
                self._fallback_to_rules()
        else:
            print("⚠️  Transformers not available, using rule-based fallback")
            self._fallback_to_rules()

    def _fallback_to_rules(self):
        """Fallback to rule-based sentiment analysis"""
        self.sentiment_pipeline = None
        print("Using rule-based sentiment analysis")

    def _load_aspect_keywords(self) -> Dict[str, List[str]]:
        """Load aspect extraction keywords"""
        return {
            'pricing': ['price', 'cost', 'expensive', 'cheap', 'affordable', 'fee', 'charge', 'money', 'payment',
                        'subscription', 'سعر', 'تكلفة', 'غالي', 'رخيص'],
            'performance': ['slow', 'fast', 'speed', 'performance', 'lag', 'quick', 'responsive', 'timeout', 'loading',
                            'بطيء', 'سريع', 'أداء'],
            'usability': ['ui', 'ux', 'interface', 'design', 'user', 'experience', 'navigation', 'menu', 'button',
                          'layout', 'واجهة', 'تصميم', 'استخدام'],
            'support': ['support', 'help', 'customer service', 'staff', 'team', 'response', 'reply', 'assistance',
                        'دعم', 'خدمة', 'مساعدة'],
            'functionality': ['bug', 'error', 'crash', 'broken', 'works', 'function', 'feature', 'issue', 'problem',
                              'خطأ', 'عطل', 'مشكلة'],
            'quality': ['quality', 'good', 'bad', 'excellent', 'terrible', 'amazing', 'awful', 'perfect', 'جودة',
                        'ممتاز', 'سيء']
        }

    def _load_sarcasm_indicators(self) -> List[str]:
        """Load sarcasm detection indicators"""
        return [
            'yeah right', 'sure thing', 'oh great', 'fantastic', 'wonderful',
            'brilliant', 'perfect', 'exactly what', 'just what', 'really helpful',
            'so helpful', 'totally', 'absolutely', '!!!', 'wow', 'amazing',
            # Arabic sarcasm indicators
            'رائع', 'ممتاز', 'تماما', 'بالضبط'
        ]

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        if not text or pd.isna(text):
            return ""

        text = str(text).strip()

        # Handle elongated words (e.g., "sooooo" -> "so")
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)

        # Convert emojis to text tokens
        text = emoji.demojize(text, delimiters=(" [", "] "))

        # Normalize URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                      '[URL]', text)

        # Normalize @mentions and hashtags
        text = re.sub(r'@[A-Za-z0-9_]+', '[USER]', text)
        text = re.sub(r'#[A-Za-z0-9_]+', '[HASHTAG]', text)

        # Basic PII removal (emails, phones)
        text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL]', text)
        text = re.sub(r'[\+]?[1-9]?[0-9]{7,15}', '[PHONE]', text)

        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def detect_language(self, text: str) -> str:
        """
        Detect language of text

        Args:
            text: Input text

        Returns:
            Detected language code
        """
        try:
            # Remove non-alphabetic characters for better detection
            clean_text = re.sub(r'[^a-zA-Zأ-ي\s]', '', text)
            if len(clean_text.strip()) < 3:
                return "en"  # Default to English for very short texts

            detected = langdetect.detect(clean_text)
            return detected
        except:
            return "en"  # Default to English if detection fails

    def detect_sarcasm(self, text: str) -> float:
        """
        Detect sarcasm indicators in text

        Args:
            text: Input text

        Returns:
            Sarcasm score (0-1)
        """
        text_lower = text.lower()
        sarcasm_score = 0.0

        # Check for sarcasm indicators
        for indicator in self.sarcasm_indicators:
            if indicator.lower() in text_lower:
                sarcasm_score += 0.2

        # Check for excessive punctuation
        exclamation_count = text.count('!')
        if exclamation_count > 2:
            sarcasm_score += 0.1 * min(exclamation_count, 5)

        # Check for conflicting sentiment words
        positive_words = ['good', 'great', 'excellent', 'perfect', 'amazing']
        negative_context = ['not', "don't", "can't", "won't", 'never']

        has_positive = any(word in text_lower for word in positive_words)
        has_negative_context = any(word in text_lower for word in negative_context)

        if has_positive and has_negative_context:
            sarcasm_score += 0.3

        return min(sarcasm_score, 1.0)

    def extract_aspects(self, text: str) -> List[Dict[str, str]]:
        """
        Extract aspects and their sentiments from text

        Args:
            text: Input text

        Returns:
            List of aspects with sentiments and rationales
        """
        aspects = []
        text_lower = text.lower()

        for aspect_name, keywords in self.aspects_keywords.items():
            aspect_mentions = []

            for keyword in keywords:
                if keyword.lower() in text_lower:
                    # Find the context around the keyword
                    start = max(0, text_lower.find(keyword.lower()) - 20)
                    end = min(len(text), text_lower.find(keyword.lower()) + len(keyword) + 20)
                    context = text[start:end]
                    aspect_mentions.append(context.strip())

            if aspect_mentions:
                # Determine aspect sentiment using simple rules
                aspect_text = ' '.join(aspect_mentions)
                aspect_sentiment = self._analyze_aspect_sentiment(aspect_text)

                aspects.append({
                    'aspect': aspect_name,
                    'sentiment': aspect_sentiment,
                    'rationale_span': aspect_text[:100] + '...' if len(aspect_text) > 100 else aspect_text
                })

        return aspects

    def _analyze_aspect_sentiment(self, text: str) -> str:
        """Analyze sentiment for a specific aspect"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'perfect', 'love', 'like', 'fantastic', 'awesome']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'horrible', 'worst', 'sucks', 'broken']

        text_lower = text.lower()

        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if negative_count > positive_count:
            return 'negative'
        elif positive_count > negative_count:
            return 'positive'
        else:
            return 'neutral'

    def _get_sentiment_scores(self, text: str) -> Dict[str, float]:
        """Get sentiment scores from model or rules"""
        if self.sentiment_pipeline:
            try:
                # Use transformer model
                results = self.sentiment_pipeline(text[:512])  # Limit text length

                # Convert to standard format
                scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}

                if isinstance(results[0], list):
                    # Handle multiple labels
                    for result in results[0]:
                        label = result['label'].lower()
                        score = result['score']

                        if 'pos' in label or label in ['5 stars', '4 stars']:
                            scores['positive'] = max(scores['positive'], score)
                        elif 'neg' in label or label in ['1 star', '2 stars']:
                            scores['negative'] = max(scores['negative'], score)
                        else:
                            scores['neutral'] = max(scores['neutral'], score)

                # Normalize scores
                total = sum(scores.values())
                if total > 0:
                    scores = {k: v / total for k, v in scores.items()}

                return scores

            except Exception as e:
                print(f"Model prediction failed: {e}, falling back to rules")

        # Fallback to rule-based sentiment
        return self._rule_based_sentiment(text)

    def _rule_based_sentiment(self, text: str) -> Dict[str, float]:
        """Rule-based sentiment analysis fallback"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity

            if polarity > 0.1:
                return {'positive': 0.6 + polarity * 0.4, 'negative': 0.1, 'neutral': 0.3 - polarity * 0.2}
            elif polarity < -0.1:
                return {'positive': 0.1, 'negative': 0.6 - polarity * 0.4, 'neutral': 0.3 + polarity * 0.2}
            else:
                return {'positive': 0.25, 'negative': 0.25, 'neutral': 0.5}
        except:
            # Most basic fallback
            return {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}

    def _apply_channel_adjustment(self, scores: Dict[str, float], channel: str) -> Dict[str, float]:
        """Apply channel-specific adjustments to sentiment scores"""
        weight = self.channel_weights.get(channel, 1.0)

        if weight != 1.0:
            # Adjust confidence based on channel reliability
            for sentiment in scores:
                if sentiment != 'neutral':
                    scores[sentiment] *= weight

            # Renormalize
            total = sum(scores.values())
            if total > 0:
                scores = {k: v / total for k, v in scores.items()}

        return scores

    def _calculate_priority(self, sentiment: str, confidence: float, aspects: List[Dict], channel: str) -> str:
        """Calculate business priority based on sentiment and context"""

        # High priority conditions
        if sentiment == 'negative' and confidence > 0.8:
            return 'high'

        # Check for high-impact aspects
        high_impact_aspects = ['support', 'functionality', 'pricing']
        negative_high_impact = any(
            aspect['aspect'] in high_impact_aspects and aspect['sentiment'] == 'negative'
            for aspect in aspects
        )

        if negative_high_impact:
            return 'high'

        # Channel-specific priorities
        if channel in ['support', 'email'] and sentiment == 'negative':
            return 'high'

        # Medium priority conditions
        if sentiment == 'negative' or (sentiment == 'neutral' and confidence < 0.6):
            return 'medium'

        return 'low'

    def _apply_temperature_scaling(self, scores: Dict[str, float], temperature: float = 1.5) -> Dict[str, float]:
        """Apply temperature scaling for calibration"""
        scaled_scores = {}

        for sentiment, score in scores.items():
            # Convert to logit, scale, convert back
            logit = np.log(max(score, 1e-7) / (1 - min(score, 1 - 1e-7)))
            scaled_logit = logit / temperature
            scaled_score = 1 / (1 + np.exp(-scaled_logit))
            scaled_scores[sentiment] = scaled_score

        # Renormalize
        total = sum(scaled_scores.values())
        if total > 0:
            scaled_scores = {k: v / total for k, v in scaled_scores.items()}

        return scaled_scores

    def predict(self, text: str, channel: str = "unknown", language: str = "auto") -> Dict[str, Any]:
        """
        Predict sentiment for input text

        Args:
            text: Input text to analyze
            channel: Source channel (reviews, social, support, etc.)
            language: Language hint or 'auto' for detection

        Returns:
            Dictionary with sentiment analysis results
        """
        if not text or not text.strip():
            return {
                "sentiment_label": "neutral",
                "confidence": 0.0,
                "aspects": [],
                "explanation": "Empty or invalid input text",
                "language_detected": "unknown",
                "channel": channel,
                "priority": "low"
            }

        # Preprocess text
        processed_text = self.preprocess_text(text)
        if len(processed_text) < 3:
            return {
                "sentiment_label": "neutral",
                "confidence": 0.5,
                "aspects": [],
                "explanation": "Text too short for reliable analysis",
                "language_detected": "unknown",
                "channel": channel,
                "priority": "low"
            }

        # Detect language
        if language == "auto":
            detected_language = self.detect_language(processed_text)
        else:
            detected_language = language

        # Get sentiment scores
        scores = self._get_sentiment_scores(processed_text)

        # Apply channel adjustments
        scores = self._apply_channel_adjustment(scores, channel)

        # Check for sarcasm and adjust confidence
        sarcasm_score = self.detect_sarcasm(processed_text)

        # Apply temperature scaling for calibration
        scores = self._apply_temperature_scaling(scores)

        # Get final prediction
        sentiment_label = max(scores.keys(), key=lambda k: scores[k])
        confidence = scores[sentiment_label]

        # Reduce confidence if sarcasm detected
        if sarcasm_score > 0.3:
            confidence *= (1 - sarcasm_score * 0.5)

        # Extract aspects
        aspects = self.extract_aspects(processed_text)

        # Calculate priority
        priority = self._calculate_priority(sentiment_label, confidence, aspects, channel)

        # Generate explanation
        explanation = self._generate_explanation(sentiment_label, confidence, sarcasm_score, aspects)

        return {
            "sentiment_label": sentiment_label,
            "confidence": float(confidence),
            "aspects": aspects,
            "explanation": explanation,
            "language_detected": detected_language,
            "channel": channel,
            "priority": priority
        }

    def _generate_explanation(self, sentiment: str, confidence: float, sarcasm_score: float,
                              aspects: List[Dict]) -> str:
        """Generate human-readable explanation for the prediction"""

        explanations = []

        # Main sentiment explanation
        if confidence > 0.8:
            explanations.append(f"Strong {sentiment} sentiment detected with high confidence.")
        elif confidence > 0.6:
            explanations.append(f"Moderate {sentiment} sentiment detected.")
        else:
            explanations.append(f"Weak {sentiment} sentiment detected with low confidence.")

        # Sarcasm warning
        if sarcasm_score > 0.3:
            explanations.append("Potential sarcasm detected, confidence adjusted downward.")

        # Aspect summary
        if aspects:
            negative_aspects = [a['aspect'] for a in aspects if a['sentiment'] == 'negative']
            if negative_aspects:
                explanations.append(f"Negative aspects identified: {', '.join(negative_aspects)}.")

        return " ".join(explanations)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "model_type": "transformer" if self.sentiment_pipeline else "rule-based",
            "supports_multilingual": True,
            "supports_aspects": True,
            "supports_calibration": True,
            "version": "1.0.0"
        }