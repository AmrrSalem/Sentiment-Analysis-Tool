# 🎯 Sentiment Analysis Tool

A production-grade sentiment analysis system designed for **messy, multi-channel customer feedback** with robust handling of multilingual text, sarcasm, misspellings, and various input quality issues.

## ✨ Features

### 🔍 **Robust Analysis**
- **Multi-class sentiment**: Positive, Negative, Neutral classification
- **Aspect extraction**: Automatically identifies pricing, UX, performance, support, etc.
- **Multilingual support**: English + Arabic with automatic language detection
- **Sarcasm detection**: Identifies sarcastic content and adjusts confidence
- **Noise handling**: Robust to misspellings, slang, emojis, and code-switching

### 📊 **Production Quality**
- **Calibrated confidence**: Temperature-scaled probabilities reflect true accuracy
- **Comprehensive metrics**: Accuracy, Precision/Recall/F1, AUROC, ECE (Expected Calibration Error)
- **Business priority**: Low/Medium/High priority scoring for actionable insights
- **Multi-channel support**: Handles reviews, social, support tickets, NPS, email, chat

### 🚀 **Easy Deployment**
- **Streamlit web app**: Interactive UI for single/batch analysis
- **REST API**: `/predict` and `/predict_batch` endpoints for integration
- **Export functionality**: Download results as CSV
- **Real-time insights**: Sentiment distribution, trending issues, time series

## 🛠️ Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd sentiment-analysis-tool

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

The app will be available at `http://localhost:8501`

API endpoints will be available at:
- `POST http://localhost:8001/predict`
- `POST http://localhost:8001/predict_batch`

### Docker (Optional)

```bash
# Build container
docker build -t sentiment-analyzer .

# Run container
docker run -p 8501:8501 -p 8001:8001 sentiment-analyzer
```

## 📖 Usage

### Web Interface

1. **Single Text Analysis**
   - Enter customer feedback in the text area
   - Select channel (reviews, social, support, etc.) and language
   - Click "Analyze Sentiment" to get detailed results

2. **Batch Analysis**
   - Upload CSV file with `text` column (optional: `channel`, `language`, `timestamp`)
   - Process multiple items with progress tracking
   - Download results with all predictions and metrics

3. **Insights Dashboard**
   - View sentiment distribution across your data
   - Identify top negative aspects and trending issues
   - Monitor sentiment over time (if timestamps provided)

4. **Model Performance**
   - View accuracy, precision, recall, F1 scores
   - Check calibration with reliability diagrams
   - Understand model strengths and limitations

### API Usage

#### Single Prediction

```python
import requests

response = requests.post("http://localhost:8001/predict", json={
    "text": "This product is amazing! 😍",
    "channel": "reviews",
    "language": "en"
})

result = response.json()
print(result)
```

#### Batch Prediction

```python
response = requests.post("http://localhost:8001/predict_batch", json={
    "texts": ["Great product!", "Terrible experience", "It's okay"],
    "channels": ["reviews", "support", "social"],
    "languages": ["en", "en", "en"]
})

results = response.json()
```

### Expected Output Format

```json
{
  "sentiment_label": "positive",
  "confidence": 0.87,
  "aspects": [
    {
      "aspect": "quality",
      "sentiment": "positive",
      "rationale_span": "amazing product quality"
    }
  ],
  "explanation": "Strong positive sentiment detected with high confidence.",
  "language_detected": "en",
  "channel": "reviews",
  "priority": "low"
}
```

## 📁 Project Structure

```
sentiment-analysis-tool/
├── app.py                    # Main Streamlit application
├── model.py                  # Sentiment analysis model & preprocessing  
├── metrics.py                # Performance metrics calculation
├── test_sentiment.py         # Unit tests
├── requirements.txt          # Python dependencies
├── data/
│   └── sample_validation.csv # Validation dataset
└── README.md                # This file
```

## 🎯 Model Architecture

### Preprocessing Pipeline
1. **Text normalization**: Handle elongated words, emojis, URLs, mentions
2. **PII removal**: Strip emails, phone numbers for privacy
3. **Multilingual processing**: Detect language, handle code-switching
4. **Sarcasm detection**: Identify conflicting sentiment indicators

### Sentiment Classification
- **Primary**: Multilingual BERT-based transformer model
- **Fallback**: Rule-based analysis using TextBlob for robustness
- **Calibration**: Temperature scaling for reliable confidence scores
- **Channel adjustment**: Context-aware confidence weighting

### Aspect Extraction
- **Keyword-based extraction** for 6 key aspects:
  - **Pricing**: cost, expensive, affordable, fees
  - **Performance**: speed, slow, fast, responsive
  - **Usability**: UI, UX, design, navigation
  - **Support**: customer service, help, staff
  - **Functionality**: bugs, errors, features
  - **Quality**: good, bad, excellent, terrible

## 📈 Performance Metrics

The system reports comprehensive performance metrics:

- **Accuracy**: Overall classification accuracy
- **Per-class Precision/Recall/F1**: Performance for each sentiment class
- **Macro F1**: Averaged F1 score across classes
- **AUROC**: Area under ROC curve (One-vs-Rest)
- **ECE**: Expected Calibration Error (confidence reliability)

### Interpretation Examples
- `"Strong performance with Macro-F1=0.84; well-calibrated (ECE=0.06)"`
- `"Good performance with Macro-F1=0.76; struggles with neutral class (F1=0.65); reasonably calibrated (ECE=0.08)"`

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_sentiment.py
```

Tests cover:
- Text preprocessing functions
- Output schema validation  
- Language detection
- Sarcasm detection
- Aspect extraction
- Edge cases (empty text, multilingual)
- Metrics computation

## ⚙️ Configuration

### Model Settings
- **Confidence threshold**: Minimum confidence for predictions
- **Aspect extraction**: Enable/disable aspect analysis
- **Language detection**: Auto-detect or specify language
- **Channel context**: Adjust for different input sources

### Supported Languages
- **English** (`en`): Full feature support
- **Arabic** (`ar`): Basic sentiment + aspect keywords
- **Mixed** (`en+ar`): Code-switching support
- **Auto-detect**: Automatic language identification

### Supported Channels
- `reviews`: Product/service reviews
- `social`: Social media posts  
- `support`: Customer support tickets
- `email`: Email communications
- `nps`: Net Promoter Score verbatims
- `chat`: Live chat transcripts
- `unknown`: Default/unspecified

## 🔧 Troubleshooting

### Common Issues

1. **Model Loading Failed**
   - Transformer models require significant memory
   - Falls back to rule-based analysis automatically
   - Consider using CPU-only mode for deployment

2. **Empty Results**
   - Check text preprocessing (may be filtering out content)
   - Verify input text length (minimum 3 characters)
   - Review language detection results

3. **Low Confidence Scores**
   - May indicate genuine uncertainty (neutral sentiment)
   - Check for sarcasm detection (reduces confidence)
   - Consider channel-specific adjustments

4. **Memory Issues**
   - Reduce batch size for large datasets
   - Use CPU-only inference
   - Consider text length limits (512 tokens)

### Performance Optimization

- **CPU vs GPU**: Configured for CPU deployment by default
- **Batch processing**: Use batch API for multiple items
- **Memory management**: Automatic cleanup between requests
- **Caching**: Results cached during session

## 🚀 Deployment Considerations

### Production Deployment
- Set appropriate memory limits (2GB+ recommended)
- Configure logging for monitoring
- Implement rate limiting for API endpoints
- Set up health checks for model availability

### Scaling
- API can be deployed separately from Streamlit UI
- Consider async processing for large batches
- Implement queue system for high-volume scenarios
- Cache frequently analyzed content

### Privacy & Security
- PII automatically stripped from text
- No data stored between requests
- Consider additional anonymization for sensitive content
- Implement authentication for production APIs

## 📝 License

This project is provided as-is for educational and commercial use. Please ensure compliance with any third-party model licenses.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Run tests (`python test_sentiment.py`)
4. Submit a pull request

## 📞 Support

For issues, questions, or feature requests:
1. Check the troubleshooting section above
2. Review test output for debugging information
3. Open an issue with detailed reproduction steps
4. Include sample input/output for faster resolution

---

**Built with ❤️ for robust, production-ready sentiment analysis**
