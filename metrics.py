import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

class MetricsCalculator:
    def __init__(self, validation_data_path: str = "data/sample_validation.csv"):
        """
        Initialize metrics calculator with validation data
        
        Args:
            validation_data_path: Path to validation dataset
        """
        self.validation_data_path = validation_data_path
        self.validation_data = None
        self.label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.reverse_label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        
        # Load validation data
        self._load_validation_data()
    
    def _load_validation_data(self):
        """Load or create validation dataset"""
        try:
            self.validation_data = pd.read_csv(self.validation_data_path)
            print(f"✅ Loaded validation data: {len(self.validation_data)} samples")
        except FileNotFoundError:
            print("⚠️ Validation data not found, creating synthetic dataset...")
            self.validation_data = self._create_synthetic_validation_data()
            self._save_validation_data()
    
    def _create_synthetic_validation_data(self) -> pd.DataFrame:
        """Create synthetic validation dataset with messy, multilingual data"""
        
        # Sample texts with known sentiments (messy, multilingual, with noise)
        validation_samples = [
            # Positive samples
            {"text": "I love this app! It's soooo good 😍", "true_label": "positive", "channel": "reviews", "language": "en"},
            {"text": "Amazing experience, highly recommend!!!", "true_label": "positive", "channel": "social", "language": "en"},
            {"text": "Perfect service, تجربة رائعة", "true_label": "positive", "channel": "support", "language": "en+ar"},
            {"text": "Great job team! Keep it up 👍", "true_label": "positive", "channel": "email", "language": "en"},
            {"text": "Excellent quality and fast delivery", "true_label": "positive", "channel": "reviews", "language": "en"},
            {"text": "This is exactly what I needed", "true_label": "positive", "channel": "chat", "language": "en"},
            {"text": "Outstanding customer support!", "true_label": "positive", "channel": "support", "language": "en"},
            {"text": "Best purchase ever made", "true_label": "positive", "channel": "reviews", "language": "en"},
            {"text": "Works perfectly, no issues", "true_label": "positive", "channel": "email", "language": "en"},
            {"text": "Very satisfied with the results", "true_label": "positive", "channel": "nps", "language": "en"},
            
            # Negative samples
            {"text": "This is terrible! Waste of money 😡", "true_label": "negative", "channel": "reviews", "language": "en"},
            {"text": "Worst experience ever, never again", "true_label": "negative", "channel": "social", "language": "en"},
            {"text": "Broken functionality, doesn't work", "true_label": "negative", "channel": "support", "language": "en"},
            {"text": "Very disappointed, سيء جداً", "true_label": "negative", "channel": "email", "language": "en+ar"},
            {"text": "Slow performance and bugs everywhere", "true_label": "negative", "channel": "reviews", "language": "en"},
            {"text": "Customer service is horrible", "true_label": "negative", "channel": "support", "language": "en"},
            {"text": "Too expensive for what you get", "true_label": "negative", "channel": "chat", "language": "en"},
            {"text": "Interface is confusing and ugly", "true_label": "negative", "channel": "reviews", "language": "en"},
            {"text": "Constantly crashes, unusable", "true_label": "negative", "channel": "support", "language": "en"},
            {"text": "Regret buying this product", "true_label": "negative", "channel": "nps", "language": "en"},
            
            # Neutral samples
            {"text": "It's okay, nothing special", "true_label": "neutral", "channel": "reviews", "language": "en"},
            {"text": "Average product, does the job", "true_label": "neutral", "channel": "social", "language": "en"},
            {"text": "Could be better, could be worse", "true_label": "neutral", "channel": "email", "language": "en"},
            {"text": "Standard quality, as expected", "true_label": "neutral", "channel": "reviews", "language": "en"},
            {"text": "No major complaints or praise", "true_label": "neutral", "channel": "chat", "language": "en"},
            {"text": "It works, that's about it", "true_label": "neutral", "channel": "support", "language": "en"},
            {"text": "Neither good nor bad", "true_label": "neutral", "channel": "nps", "language": "en"},
            {"text": "Decent but room for improvement", "true_label": "neutral", "channel": "reviews", "language": "en"},
            {"text": "Meets basic requirements only", "true_label": "neutral", "channel": "email", "language": "en"},
            {"text": "Mixed feelings about this", "true_label": "neutral", "channel": "social", "language": "en"},
            
            # Sarcastic samples (challenging cases)
            {"text": "Oh great, another bug! Just perfect 🙄", "true_label": "negative", "channel": "support", "language": "en"},
            {"text": "Yeah right, 'excellent' service", "true_label": "negative", "channel": "reviews", "language": "en"},
            {"text": "Totally amazing how it never works", "true_label": "negative", "channel": "social", "language": "en"},
            
            # Misspelled samples
            {"text": "Realy gud prodct, luv it", "true_label": "positive", "channel": "reviews", "language": "en"},
            {"text": "Horible experiance, vry bad", "true_label": "negative", "channel": "social", "language": "en"},
            
            # Short samples
            {"text": "Good", "true_label": "positive", "channel": "nps", "language": "en"},
            {"text": "Bad", "true_label": "negative", "channel": "nps", "language": "en"},
            {"text": "OK", "true_label": "neutral", "channel": "nps", "language": "en"},
            
            # Mixed language samples
            {"text": "Very good جيد جداً excellent", "true_label": "positive", "channel": "reviews", "language": "en+ar"},
            {"text": "Bad service خدمة سيئة", "true_label": "negative", "channel": "support", "language": "en+ar"},
        ]
        
        return pd.DataFrame(validation_samples)
    
    def _save_validation_data(self):
        """Save validation data to file"""
        import os
        os.makedirs("data", exist_ok=True)
        self.validation_data.to_csv(self.validation_data_path, index=False)
        print(f"💾 Saved validation data to {self.validation_data_path}")
    
    def get_validation_data(self) -> pd.DataFrame:
        """Get the validation dataset"""
        return self.validation_data.copy()
    
    def _get_predictions(self, analyzer) -> Tuple[List[int], List[List[float]], List[int]]:
        """Get predictions from analyzer for validation data"""
        y_true = []
        y_proba = []
        y_pred = []
        
        for _, row in self.validation_data.iterrows():
            text = row['text']
            true_label = row['true_label']
            channel = row.get('channel', 'unknown')
            language = row.get('language', 'auto')
            
            # Get prediction
            result = analyzer.predict(text=text, channel=channel, language=language)
            
            # Convert labels to numeric
            y_true.append(self.label_map[true_label])
            y_pred.append(self.label_map[result['sentiment_label']])
            
            # Get probability scores
            pred_sentiment = result['sentiment_label']
            confidence = result['confidence']
            
            proba = [0.33, 0.34, 0.33]  # Default equal probabilities
            pred_idx = self.label_map[pred_sentiment]
            proba[pred_idx] = confidence
            
            # Redistribute remaining probability
            remaining = (1 - confidence) / 2
            for i in range(3):
                if i != pred_idx:
                    proba[i] = remaining
            
            y_proba.append(proba)
        
        return y_true, y_proba, y_pred
    
    def compute_accuracy(self, y_true: List[int], y_pred: List[int]) -> float:
        """Compute accuracy score"""
        return accuracy_score(y_true, y_pred)
    
    def compute_precision_recall_f1(self, y_true: List[int], y_pred: List[int]) -> Dict[str, Dict[str, float]]:
        """Compute precision, recall, and F1 scores per class and macro averages"""
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0
        )
        
        # Macro averages
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        
        # Per-class results
        per_class = {}
        for i, label in enumerate(['negative', 'neutral', 'positive']):
            per_class[label] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            }
        
        return {
            'per_class': per_class,
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall),
            'macro_f1': float(macro_f1)
        }
    
    def compute_auroc(self, y_true: List[int], y_proba: List[List[float]]) -> float:
        """Compute AUROC (One-vs-Rest)"""
        try:
            # Binarize labels for multiclass AUROC
            y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
            y_proba_array = np.array(y_proba)
            
            # Handle case where not all classes are present
            if y_true_bin.shape[1] < 3:
                return 0.5  # Random performance
            
            auroc = roc_auc_score(y_true_bin, y_proba_array, average='macro', multi_class='ovr')
            return float(auroc)
        except Exception as e:
            print(f"Error computing AUROC: {e}")
            return 0.5
    
    def compute_ece(self, y_true: List[int], y_proba: List[List[float]], y_pred: List[int], n_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error (ECE)
        """
        try:
            # Get confidence scores (max probability)
            confidences = [max(proba) for proba in y_proba]
            accuracies = [1 if yt == yp else 0 for yt, yp in zip(y_true, y_pred)]
            
            # Create bins
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            total_samples = len(y_true)
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                # Find samples in this bin
                in_bin = [(conf > bin_lower) and (conf <= bin_upper) for conf in confidences]
                prop_in_bin = sum(in_bin) / total_samples if total_samples > 0 else 0
                
                if prop_in_bin > 0:
                    # Compute accuracy and confidence in this bin
                    bin_accuracies = [acc for acc, in_b in zip(accuracies, in_bin) if in_b]
                    bin_confidences = [conf for conf, in_b in zip(confidences, in_bin) if in_b]
                    
                    accuracy_in_bin = np.mean(bin_accuracies) if bin_accuracies else 0
                    avg_confidence_in_bin = np.mean(bin_confidences) if bin_confidences else 0
                    
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            return float(ece)
        
        except Exception as e:
            print(f"Error computing ECE: {e}")
            return 0.0
    
    def compute_all_metrics(self, analyzer) -> Dict[str, Any]:
        """Compute all performance metrics"""
        
        print("Computing model performance metrics...")
        
        # Get predictions
        y_true, y_proba, y_pred = self._get_predictions(analyzer)
        
        # Compute individual metrics
        accuracy = self.compute_accuracy(y_true, y_pred)
        pr_f1_results = self.compute_precision_recall_f1(y_true, y_pred)
        auroc = self.compute_auroc(y_true, y_proba)
        ece = self.compute_ece(y_true, y_proba, y_pred)
        
        # Generate interpretation
        interpretation = self._generate_interpretation(
            accuracy, pr_f1_results['macro_f1'], auroc, ece, pr_f1_results['per_class']
        )
        
        return {
            'accuracy': accuracy,
            'per_class_metrics': pr_f1_results['per_class'],
            'macro_precision': pr_f1_results['macro_precision'],
            'macro_recall': pr_f1_results['macro_recall'],
            'macro_f1': pr_f1_results['macro_f1'],
            'auroc': auroc,
            'ece': ece,
            'interpretation': interpretation,
            'y_true': y_true,
            'y_proba': y_proba,
            'y_pred': y_pred
        }
    
    def _generate_interpretation(self, accuracy: float, macro_f1: float, auroc: float, 
                               ece: float, per_class: Dict) -> str:
        """Generate human-readable interpretation of metrics"""
        
        interpretations = []
        
        # Overall performance
        if macro_f1 >= 0.8:
            interpretations.append(f"Strong performance with Macro-F1={macro_f1:.3f}")
        elif macro_f1 >= 0.7:
            interpretations.append(f"Good performance with Macro-F1={macro_f1:.3f}")
        elif macro_f1 >= 0.6:
            interpretations.append(f"Moderate performance with Macro-F1={macro_f1:.3f}")
        else:
            interpretations.append(f"Needs improvement with Macro-F1={macro_f1:.3f}")
        
        # Class-specific issues
        worst_class = min(per_class.items(), key=lambda x: x[1]['f1'])
        if worst_class[1]['f1'] < 0.6:
            interpretations.append(f"struggles with {worst_class[0]} class (F1={worst_class[1]['f1']:.3f})")
        
        # Calibration assessment
        if ece <= 0.05:
            interpretations.append(f"well-calibrated (ECE={ece:.3f})")
        elif ece <= 0.10:
            interpretations.append(f"reasonably calibrated (ECE={ece:.3f})")
        else:
            confidence_direction = "over-confident" if ece > 0.15 else "under-confident"
            interpretations.append(f"{confidence_direction} (ECE={ece:.3f})")
        
        return "; ".join(interpretations) + "."
    
    def plot_reliability_diagram(self, y_true: List[int], y_proba: List[List[float]], 
                                y_pred: List[int], n_bins: int = 10):
        """Create reliability diagram (calibration plot)"""
        
        # Set style
        try:
            plt.style.use('seaborn')
        except:
            plt.style.use('default')
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Get confidence scores
        confidences = [max(proba) for proba in y_proba]
        accuracies = [1 if yt == yp else 0 for yt, yp in zip(y_true, y_pred)]
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_centers = []
        bin_accuracies = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = [(conf > bin_lower) and (conf <= bin_upper) for conf in confidences]
            
            if sum(in_bin) > 0:
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_acc = np.mean([acc for acc, in_b in zip(accuracies, in_bin) if in_b])
                bin_accuracies.append(bin_acc)
                bin_counts.append(sum(in_bin))
            else:
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(0)
                bin_counts.append(0)
        
        # Plot reliability diagram
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
        bars = ax.bar(bin_centers, bin_accuracies, width=0.08, alpha=0.7, 
                     edgecolor='black', label='Model')
        
        # Add count labels on bars
        for bar, count in zip(bars, bin_counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'n={count}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_title('Reliability Diagram (Calibration Plot)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrix(self, y_true: List[int], y_pred: List[int]):
        """Create confusion matrix plot"""
        
        # Set style
        try:
            plt.style.use('seaborn')
        except:
            plt.style.use('default')
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        
        # Create heatmap
        labels = ['Negative', 'Neutral', 'Positive']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels, ax=ax)
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        
        plt.tight_layout()
        return fig
