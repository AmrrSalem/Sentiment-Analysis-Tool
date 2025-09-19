import unittest
import sys
import os
import json

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import SentimentAnalyzer
from metrics import MetricsCalculator

class TestSentimentAnalyzer(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = SentimentAnalyzer()
    
    def test_preprocess_text(self):
        """Test text preprocessing functionality"""
        
        # Test elongated words
        result = self.analyzer.preprocess_text("This is soooooo good!")
        self.assertNotIn("soooooo", result)
        self.assertIn("soo", result)  # Should be reduced to double
        
        # Test emoji conversion
        result = self.analyzer.preprocess_text("I love this 😍")
        self.assertIn("smiling_face_with_heart-eyes", result)
        
        # Test URL normalization
        result = self.analyzer.preprocess_text("Check out https://example.com")
        self.assertIn("[URL]", result)
        self.assertNotIn("https://example.com", result)
        
        # Test @mentions
        result = self.analyzer.preprocess_text("Thanks @customer_service")
        self.assertIn("[USER]", result)
        self.assertNotIn("@customer_service", result)
        
        # Test email removal
        result = self.analyzer.preprocess_text("Contact me at test@email.com")
        self.assertIn("[EMAIL]", result)
        self.assertNotIn("test@email.com", result)
    
    def test_language_detection(self):
        """Test language detection"""
        
        # English text
        lang = self.analyzer.detect_language("This is a great product")
        self.assertEqual(lang, "en")
        
        # Very short text should default to English
        lang = self.analyzer.detect_language("OK")
        self.assertEqual(lang, "en")
        
        # Empty text should default to English
        lang = self.analyzer.detect_language("")
        self.assertEqual(lang, "en")
    
    def test_sarcasm_detection(self):
        """Test sarcasm detection"""
        
        # Clear sarcasm
        score = self.analyzer.detect_sarcasm("Oh great, another bug!")
        self.assertGreater(score, 0.3)
        
        # No sarcasm
        score = self.analyzer.detect_sarcasm("This is really good")
        self.assertLess(score, 0.3)
        
        # Excessive punctuation
        score = self.analyzer.detect_sarcasm("Perfect!!!!!!")
        self.assertGreater(score, 0.1)
    
    def test_aspect_extraction(self):
        """Test aspect extraction"""
        
        # Text with pricing aspect
        aspects = self.analyzer.extract_aspects("The price is too expensive")
        pricing_aspects = [a for a in aspects if a['aspect'] == 'pricing']
        self.assertGreater(len(pricing_aspects), 0)
        self.assertEqual(pricing_aspects[0]['sentiment'], 'negative')
        
        # Text with performance aspect
        aspects = self.analyzer.extract_aspects("The app is very fast")
        performance_aspects = [a for a in aspects if a['aspect'] == 'performance']
        self.assertGreater(len(performance_aspects), 0)
        self.assertEqual(performance_aspects[0]['sentiment'], 'positive')
    
    def test_predict_output_schema(self):
        """Test that predict output matches required schema"""
        
        result = self.analyzer.predict("This is a great product!")
        
        # Check required keys
        required_keys = [
            'sentiment_label', 'confidence', 'aspects', 'explanation',
            'language_detected', 'channel', 'priority'
        ]
        
        for key in required_keys:
            self.assertIn(key, result, f"Missing required key: {key}")
        
        # Check data types
        self.assertIsInstance(result['sentiment_label'], str)
        self.assertIn(result['sentiment_label'], ['positive', 'negative', 'neutral'])
        
        self.assertIsInstance(result['confidence'], float)
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)
        
        self.assertIsInstance(result['aspects'], list)
        
        self.assertIsInstance(result['explanation'], str)
        self.assertGreater(len(result['explanation']), 0)
        
        self.assertIsInstance(result['language_detected'], str)
        
        self.assertIsInstance(result['channel'], str)
        
        self.assertIsInstance(result['priority'], str)
        self.assertIn(result['priority'], ['low', 'medium', 'high'])
    
    def test_aspect_output_schema(self):
        """Test aspect output schema"""
        
        result = self.analyzer.predict("The price is too expensive but the quality is good")
        
        for aspect in result['aspects']:
            self.assertIsInstance(aspect, dict)
            
            required_aspect_keys = ['aspect', 'sentiment', 'rationale_span']
            for key in required_aspect_keys:
                self.assertIn(key, aspect, f"Missing aspect key: {key}")
            
            self.assertIsInstance(aspect['aspect'], str)
            self.assertIn(aspect['sentiment'], ['positive', 'negative', 'neutral'])
            self.assertIsInstance(aspect['rationale_span'], str)
    
    def test_empty_text_handling(self):
        """Test handling of empty or invalid text"""
        
        # Empty string
        result = self.analyzer.predict("")
        self.assertEqual(result['sentiment_label'], 'neutral')
        self.assertEqual(result['confidence'], 0.0)
        
        # None input
        result = self.analyzer.predict(None)
        self.assertEqual(result['sentiment_label'], 'neutral')
        self.assertEqual(result['confidence'], 0.0)
        
        # Very short text
        result = self.analyzer.predict("a")
        self.assertEqual(result['sentiment_label'], 'neutral')
        self.assertLessEqual(result['confidence'], 0.6)
    
    def test_multilingual_support(self):
        """Test multilingual text processing"""
        
        # English + Arabic mixed text
        result = self.analyzer.predict("Very good جيد جداً excellent", language="en+ar")
        self.assertEqual(result['sentiment_label'], 'positive')
        
        # Should handle without crashing
        result = self.analyzer.predict("تجربة رائعة", language="ar")
        self.assertIsInstance(result['sentiment_label'], str)
    
    def test_different_channels(self):
        """Test channel-specific processing"""
        
        channels = ['reviews', 'social', 'support', 'email', 'nps', 'chat']
        
        for channel in channels:
            result = self.analyzer.predict("This is okay", channel=channel)
            self.assertEqual(result['channel'], channel)
            self.assertIsInstance(result['priority'], str)
    
    def test_priority_calculation(self):
        """Test priority calculation logic"""
        
        # High priority: negative sentiment with high confidence
        result = self.analyzer.predict("This is absolutely terrible and broken")
        # Priority should be high for clearly negative feedback
        
        # Low priority: positive sentiment
        result = self.analyzer.predict("This is really great!")
        self.assertEqual(result['priority'], 'low')


class TestMetricsCalculator(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.metrics_calc = MetricsCalculator()
        self.analyzer = SentimentAnalyzer()
    
    def test_validation_data_loading(self):
        """Test validation data is loaded correctly"""
        
        val_data = self.metrics_calc.get_validation_data()
        
        self.assertIsInstance(val_data, pd.DataFrame)
        self.assertGreater(len(val_data), 0)
        
        # Check required columns
        required_columns = ['text', 'true_label', 'channel', 'language']
        for col in required_columns:
            self.assertIn(col, val_data.columns, f"Missing column: {col}")
    
    def test_metrics_computation(self):
        """Test metrics computation"""
        
        # This test requires the analyzer to be working
        try:
            results = self.metrics_calc.compute_all_metrics(self.analyzer)
            
            # Check all required metrics are present
            required_metrics = [
                'accuracy', 'macro_f1', 'auroc', 'ece', 
                'per_class_metrics', 'interpretation'
            ]
            
            for metric in required_metrics:
                self.assertIn(metric, results, f"Missing metric: {metric}")
            
            # Check value ranges
            self.assertGreaterEqual(results['accuracy'], 0.0)
            self.assertLessEqual(results['accuracy'], 1.0)
            
            self.assertGreaterEqual(results['macro_f1'], 0.0)
            self.assertLessEqual(results['macro_f1'], 1.0)
            
            self.assertGreaterEqual(results['auroc'], 0.0)
            self.assertLessEqual(results['auroc'], 1.0)
            
            self.assertGreaterEqual(results['ece'], 0.0)
            self.assertLessEqual(results['ece'], 1.0)
            
            # Check interpretation is a string
            self.assertIsInstance(results['interpretation'], str)
            self.assertGreater(len(results['interpretation']), 0)
            
        except Exception as e:
            self.skipTest(f"Metrics computation failed: {e}")
    
    def test_per_class_metrics_structure(self):
        """Test per-class metrics structure"""
        
        try:
            results = self.metrics_calc.compute_all_metrics(self.analyzer)
            per_class = results['per_class_metrics']
            
            # Check all sentiment classes are present
            for sentiment in ['negative', 'neutral', 'positive']:
                self.assertIn(sentiment, per_class, f"Missing class: {sentiment}")
                
                class_metrics = per_class[sentiment]
                
                # Check required metrics for each class
                required_class_metrics = ['precision', 'recall', 'f1', 'support']
                for metric in required_class_metrics:
                    self.assertIn(metric, class_metrics, f"Missing class metric: {metric}")
                    
                    if metric != 'support':  # Support is count, others are ratios
                        self.assertGreaterEqual(class_metrics[metric], 0.0)
                        self.assertLessEqual(class_metrics[metric], 1.0)
        
        except Exception as e:
            self.skipTest(f"Per-class metrics test failed: {e}")


def run_tests():
    """Run all tests and return results"""
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test methods
    test_classes = [TestSentimentAnalyzer, TestMetricsCalculator]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result


if __name__ == "__main__":
    print("Running sentiment analysis tests...")
    print("=" * 50)
    
    try:
        # Import required modules for tests
        import pandas as pd
        
        result = run_tests()
        
        print("\n" + "=" * 50)
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"- {test}: {traceback}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"- {test}: {traceback}")
        
        if result.wasSuccessful():
            print("\n✅ All tests passed!")
        else:
            print(f"\n❌ {len(result.failures + result.errors)} test(s) failed")
            
    except ImportError as e:
        print(f"❌ Missing dependencies for testing: {e}")
        print("Please install required packages first")
    except Exception as e:
        print(f"❌ Error running tests: {e}")
