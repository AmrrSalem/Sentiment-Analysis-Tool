# Sentiment Analysis Tool

A production-ready sentiment analysis system combining **transformer-based NLP models** with a real-time **Streamlit + FastAPI** interface.

## Features

- Multi-class sentiment classification (Positive / Negative / Neutral)
- Transformer models via HuggingFace (`transformers`)
- Real-time REST API with **FastAPI** + **Uvicorn**
- Interactive **Streamlit** dashboard for live predictions
- Full evaluation metrics: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
- Batch prediction support via CSV upload
- Dockerized for easy deployment

## Tech Stack

| Layer | Tools |
|-------|-------|
| NLP Model | HuggingFace Transformers, scikit-learn |
| API | FastAPI, Uvicorn, Pydantic |
| Dashboard | Streamlit |
| Metrics | scikit-learn, Matplotlib, Seaborn |

## Quick Start

```bash
pip install -r requirements.txt
python run.py
```

Then open `http://localhost:8501` for the Streamlit dashboard or `http://localhost:8000/docs` for the API.

## Project Structure

```
├── model.py        # SentimentAnalyzer class (transformers + sklearn)
├── app.py          # Streamlit dashboard + FastAPI server
├── metrics.py      # MetricsCalculator for evaluation
├── run.py          # Entry point
└── requirements.txt
```
