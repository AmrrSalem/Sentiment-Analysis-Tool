#!/usr/bin/env python3
"""
Compatibility test script to verify all components work together
"""

import sys
import os
import traceback

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        import pandas as pd
        print("✅ pandas imported")
        
        import numpy as np
        print("✅ numpy imported")
        
        import matplotlib.pyplot as plt
        print("✅ matplotlib imported")
        
        import seaborn as sns
        print("✅ seaborn imported")
        
        from sklearn.metrics import accuracy_score
        print("✅ sklearn imported")
        
        import streamlit as st
        print("✅ streamlit imported")
        
        import fastapi
        print("✅ fastapi imported")
        
        import uvicorn
        print("✅ uvicorn imported")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_loading():
    """Test sentiment analyzer loading"""
    print("\nTesting model loading...")
    
    try:
        from model import SentimentAnalyzer
        analyzer = SentimentAnalyzer()
        print("✅ SentimentAnalyzer loaded")
        
        # Test basic prediction
        result = analyzer.predict("This is a test")
        required_keys = ['sentiment_label', 'confidence', 'aspects', 'explanation']
        
        for key in required_keys:
            if key not in result:
                print(f"❌ Missing key in prediction: {key}")
                return False
        
        print(f"✅ Basic prediction works: {result['sentiment_label']}")
        return True
        
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        traceback.print_exc()
        return False

def test_metrics_loading():
    """Test metrics calculator loading"""
    print("\nTesting metrics loading...")
    
    try:
        from metrics import MetricsCalculator
        metrics_calc = MetricsCalculator()
        print("✅ MetricsCalculator loaded")
        
        val_data = metrics_calc.get_validation_data()
        if len(val_data) > 0:
            print(f"✅ Validation data loaded: {len(val_data)} samples")
        else:
            print("❌ No validation data")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Metrics loading error: {e}")
        traceback.print_exc()
        return False

def test_integration():
    """Test integration between components"""
    print("\nTesting integration...")
    
    try:
        from model import SentimentAnalyzer
        from metrics import MetricsCalculator
        
        analyzer = SentimentAnalyzer()
        metrics_calc = MetricsCalculator()
        
        # Test metrics computation (may take a moment)
        print("Computing metrics (this may take 10-30 seconds)...")
        results = metrics_calc.compute_all_metrics(analyzer)
        
        required_metrics = ['accuracy', 'macro_f1', 'auroc', 'ece']
        for metric in required_metrics:
            if metric not in results:
                print(f"❌ Missing metric: {metric}")
                return False
            print(f"✅ {metric}: {results[metric]:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration error: {e}")
        traceback.print_exc()
        return False

def test_file_structure():
    """Test required files exist"""
    print("\nTesting file structure...")
    
    required_files = ['app.py', 'model.py', 'metrics.py']
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
            return False
    
    # Check data directory
    if not os.path.exists('data'):
        print("📁 Creating data directory...")
        os.makedirs('data', exist_ok=True)
    
    print("✅ File structure OK")
    return True

def main():
    """Run all compatibility tests"""
    print("🧪 Running compatibility tests...")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("Model Loading", test_model_loading),
        ("Metrics Loading", test_metrics_loading),
        ("Integration", test_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}")
        print("-" * 30)
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Your system is ready to run.")
        print("\nTo start the app:")
        print("  python run.py")
        print("  or")
        print("  streamlit run app.py --server.port 8502")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())