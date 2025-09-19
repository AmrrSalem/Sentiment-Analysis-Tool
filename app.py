import streamlit as st
import pandas as pd
import numpy as np
import json
import threading
import time
from datetime import datetime
import io
import base64
from pathlib import Path

# Import our modules
from model import SentimentAnalyzer
from metrics import MetricsCalculator
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# Configure Streamlit
st.set_page_config(
    page_title="Sentiment Analysis Tool",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# FastAPI app for the endpoint
api_app = FastAPI(title="Sentiment Analysis API")

class PredictionRequest(BaseModel):
    text: str
    channel: Optional[str] = "unknown"
    language: Optional[str] = "auto"

class BatchPredictionRequest(BaseModel):
    texts: List[str]
    channels: Optional[List[str]] = None
    languages: Optional[List[str]] = None

@api_app.post("/predict")
async def predict_sentiment(request: PredictionRequest):
    try:
        analyzer = st.session_state.get('analyzer')
        if analyzer is None:
            raise HTTPException(status_code=500, detail="Model not initialized")
        
        result = analyzer.predict(
            text=request.text,
            channel=request.channel,
            language=request.language
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_app.post("/predict_batch")
async def predict_batch(request: BatchPredictionRequest):
    try:
        analyzer = st.session_state.get('analyzer')
        if analyzer is None:
            raise HTTPException(status_code=500, detail="Model not initialized")
        
        results = []
        channels = request.channels or ["unknown"] * len(request.texts)
        languages = request.languages or ["auto"] * len(request.texts)
        
        for i, text in enumerate(request.texts):
            channel = channels[i] if i < len(channels) else "unknown"
            language = languages[i] if i < len(languages) else "auto"
            
            result = analyzer.predict(text=text, channel=channel, language=language)
            results.append(result)
        
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Initialize session state
def initialize_session_state():
    if 'analyzer' not in st.session_state:
        with st.spinner('Loading sentiment analysis model...'):
            st.session_state.analyzer = SentimentAnalyzer()
    
    if 'metrics_calculator' not in st.session_state:
        st.session_state.metrics_calculator = MetricsCalculator()
    
    if 'api_server' not in st.session_state:
        st.session_state.api_server = None

def start_api_server():
    """Start FastAPI server in a separate thread"""
    if st.session_state.api_server is None:
        def run_server():
            uvicorn.run(api_app, host="0.0.0.0", port=8001, log_level="error")
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        st.session_state.api_server = server_thread
        time.sleep(1)  # Give server time to start

def main():
    st.title("🎯 Sentiment Analysis Tool")
    st.markdown("**Production-grade sentiment analysis for messy, multi-channel customer feedback**")
    
    # Initialize components
    initialize_session_state()
    start_api_server()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model settings
        st.subheader("Model Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5, 
            step=0.05,
            help="Minimum confidence for predictions"
        )
        
        enable_aspect_extraction = st.checkbox(
            "Enable Aspect Extraction", 
            value=True,
            help="Extract specific aspects (pricing, UX, etc.)"
        )
        
        # Language settings
        st.subheader("Language Settings")
        supported_languages = ["auto", "en", "ar", "en+ar"]
        default_language = st.selectbox(
            "Default Language Detection",
            options=supported_languages,
            index=0,
            help="Language for processing text"
        )
        
        # Channel settings
        st.subheader("Channel Settings")
        supported_channels = ["unknown", "reviews", "social", "support", "email", "nps", "chat"]
        default_channel = st.selectbox(
            "Default Channel",
            options=supported_channels,
            index=0,
            help="Source channel for context"
        )
        
        # API endpoint info
        st.subheader("🔗 API Endpoint")
        st.code("POST http://localhost:8001/predict", language="bash")
        st.code("POST http://localhost:8001/predict_batch", language="bash")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Insights", "📈 Response Accuracy & Quality"])
    
    with tab1:
        predict_tab(confidence_threshold, enable_aspect_extraction, default_language, default_channel)
    
    with tab2:
        insights_tab()
    
    with tab3:
        accuracy_tab()

def predict_tab(confidence_threshold, enable_aspect_extraction, default_language, default_channel):
    st.header("Sentiment Prediction")
    
    # Single prediction
    st.subheader("Single Text Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        input_text = st.text_area(
            "Enter text for sentiment analysis:",
            placeholder="Type your customer feedback here...",
            height=100
        )
    
    with col2:
        channel = st.selectbox("Channel:", ["auto"] + ["reviews", "social", "support", "email", "nps", "chat"], key="single_channel")
        language = st.selectbox("Language:", ["auto", "en", "ar", "en+ar"], key="single_lang")
    
    if st.button("Analyze Sentiment", type="primary"):
        if input_text.strip():
            with st.spinner("Analyzing..."):
                result = st.session_state.analyzer.predict(
                    text=input_text,
                    channel=channel if channel != "auto" else default_channel,
                    language=language if language != "auto" else default_language
                )
                
                display_single_result(result, confidence_threshold)
        else:
            st.error("Please enter some text to analyze.")
    
    st.divider()
    
    # Batch prediction
    st.subheader("Batch Analysis")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="CSV should contain a 'text' column. Optional: 'channel', 'language', 'timestamp' columns"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            if 'text' not in df.columns:
                st.error("CSV file must contain a 'text' column")
                return
            
            st.write(f"📄 Loaded {len(df)} rows")
            st.write("Preview:")
            st.dataframe(df.head(), use_container_width=True)
            
            if st.button("Analyze Batch", type="primary"):
                with st.spinner("Processing batch..."):
                    results = process_batch(df, default_language, default_channel, enable_aspect_extraction)
                    
                    # Display results
                    st.success(f"✅ Processed {len(results)} items")
                    
                    # Convert results to DataFrame
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download button
                    csv_buffer = io.StringIO()
                    results_df.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    
                    st.download_button(
                        label="📥 Download Results",
                        data=csv_data,
                        file_name=f"sentiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Store results for insights
                    st.session_state.batch_results = results_df
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def display_single_result(result, confidence_threshold):
    """Display single prediction result with formatting"""
    
    # Main sentiment with confidence
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        sentiment = result['sentiment_label']
        emoji_map = {'positive': '😊', 'negative': '😞', 'neutral': '😐'}
        st.metric(
            "Sentiment",
            f"{emoji_map.get(sentiment, '🤔')} {sentiment.title()}",
            help="Predicted sentiment category"
        )
    
    with col2:
        confidence = result['confidence']
        color = "green" if confidence >= confidence_threshold else "orange"
        st.metric(
            "Confidence",
            f"{confidence:.2%}",
            help=f"Model confidence (threshold: {confidence_threshold:.0%})"
        )
    
    with col3:
        priority = result.get('priority', 'medium')
        priority_colors = {'low': '🟢', 'medium': '🟡', 'high': '🔴'}
        st.metric(
            "Priority",
            f"{priority_colors.get(priority, '⚪')} {priority.title()}",
            help="Business priority level"
        )
    
    # Explanation
    st.write("**Explanation:**", result.get('explanation', 'No explanation available'))
    
    # Detected language and channel
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Language:** {result.get('language_detected', 'unknown')}")
    with col2:
        st.info(f"**Channel:** {result.get('channel', 'unknown')}")
    
    # Aspects
    aspects = result.get('aspects', [])
    if aspects:
        st.write("**Extracted Aspects:**")
        
        for aspect in aspects:
            aspect_sentiment = aspect.get('sentiment', 'neutral')
            aspect_name = aspect.get('aspect', 'unknown')
            rationale = aspect.get('rationale_span', 'No rationale')
            
            emoji_map = {'positive': '✅', 'negative': '❌', 'neutral': '➖'}
            
            with st.expander(f"{emoji_map.get(aspect_sentiment, '❓')} {aspect_name.title()} ({aspect_sentiment})"):
                st.write(f"**Rationale:** {rationale}")
    
    # Raw JSON (collapsible)
    with st.expander("🔍 Raw JSON Output"):
        st.json(result)

def process_batch(df, default_language, default_channel, enable_aspect_extraction):
    """Process batch of texts"""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, row in df.iterrows():
        text = row.get('text', '')
        channel = row.get('channel', default_channel)
        language = row.get('language', default_language)
        
        if text and text.strip():
            try:
                result = st.session_state.analyzer.predict(
                    text=text,
                    channel=channel,
                    language=language
                )
                
                # Add original text and any additional columns
                result['original_text'] = text
                if 'timestamp' in row:
                    result['timestamp'] = row['timestamp']
                
                results.append(result)
                
            except Exception as e:
                st.warning(f"Error processing row {i}: {str(e)}")
                continue
        
        # Update progress
        progress = (i + 1) / len(df)
        progress_bar.progress(progress)
        status_text.text(f"Processing {i+1}/{len(df)} items...")
    
    progress_bar.empty()
    status_text.empty()
    
    return results

def insights_tab():
    st.header("📊 Insights Dashboard")
    
    if 'batch_results' not in st.session_state or st.session_state.batch_results is None:
        st.info("Upload and analyze a batch of data in the Predict tab to see insights.")
        return
    
    df = st.session_state.batch_results
    
    if df.empty:
        st.warning("No data available for insights.")
        return
    
    # Sentiment distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sentiment Distribution")
        sentiment_counts = df['sentiment_label'].value_counts()
        st.bar_chart(sentiment_counts)
        
        # Show percentages
        total = len(df)
        for sentiment, count in sentiment_counts.items():
            st.write(f"**{sentiment.title()}:** {count} ({count/total:.1%})")
    
    with col2:
        st.subheader("Priority Distribution")
        if 'priority' in df.columns:
            priority_counts = df['priority'].value_counts()
            st.bar_chart(priority_counts)
        else:
            st.info("No priority data available")
    
    # Top negative aspects
    st.subheader("Top Issues (Negative Aspects)")
    
    negative_aspects = []
    for _, row in df.iterrows():
        aspects = row.get('aspects', [])
        if isinstance(aspects, str):
            try:
                aspects = json.loads(aspects)
            except:
                continue
        
        if isinstance(aspects, list):
            for aspect in aspects:
                if isinstance(aspect, dict) and aspect.get('sentiment') == 'negative':
                    negative_aspects.append(aspect.get('aspect', 'unknown'))
    
    if negative_aspects:
        aspect_counts = pd.Series(negative_aspects).value_counts().head(10)
        st.bar_chart(aspect_counts)
    else:
        st.info("No negative aspects found in the data.")
    
    # Time trends (if timestamp available)
    if 'timestamp' in df.columns:
        st.subheader("Sentiment Trends Over Time")
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df_time = df.groupby([df['timestamp'].dt.date, 'sentiment_label']).size().unstack(fill_value=0)
            st.line_chart(df_time)
        except Exception as e:
            st.error(f"Error creating time trends: {str(e)}")

def accuracy_tab():
    st.header("📈 Response Accuracy & Quality")
    
    # Load validation data and compute metrics
    metrics_calc = st.session_state.metrics_calculator
    
    with st.spinner("Computing model performance metrics..."):
        metrics_results = metrics_calc.compute_all_metrics(st.session_state.analyzer)
    
    # Display metrics
    st.subheader("Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{metrics_results['accuracy']:.3f}")
    
    with col2:
        st.metric("Macro F1", f"{metrics_results['macro_f1']:.3f}")
    
    with col3:
        st.metric("AUROC", f"{metrics_results['auroc']:.3f}")
    
    with col4:
        st.metric("ECE", f"{metrics_results['ece']:.3f}", help="Expected Calibration Error")
    
    # Per-class metrics
    st.subheader("Per-Class Performance")
    
    class_metrics_df = pd.DataFrame(metrics_results['per_class_metrics']).T
    st.dataframe(class_metrics_df, use_container_width=True)
    
    # Interpretation
    st.subheader("Model Interpretation")
    st.info(metrics_results['interpretation'])
    
    # Reliability diagram
    st.subheader("Calibration Plot")
    fig = metrics_calc.plot_reliability_diagram(
        metrics_results['y_true'], 
        metrics_results['y_proba'], 
        metrics_results['y_pred']
    )
    st.pyplot(fig)
    
    # Confusion matrix
    st.subheader("Confusion Matrix")
    fig_cm = metrics_calc.plot_confusion_matrix(
        metrics_results['y_true'], 
        metrics_results['y_pred']
    )
    st.pyplot(fig_cm)
    
    # Validation data sample
    with st.expander("🔍 Validation Data Sample"):
        val_df = metrics_calc.get_validation_data().head(10)
        st.dataframe(val_df, use_container_width=True)

if __name__ == "__main__":
    main()
