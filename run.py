#!/usr/bin/env python3
"""
Startup script for Sentiment Analysis Tool
Ensures proper directory structure and launches the app
"""

import os
import sys
import subprocess
from pathlib import Path


def setup_directories():
    """Create necessary directories"""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    print(f"✅ Created/verified data directory: {data_dir}")


def check_dependencies():
    """Check if required dependencies are available"""
    required_modules = [
        'streamlit', 'pandas', 'numpy', 'sklearn',
        'textblob', 'langdetect', 'emoji', 'matplotlib',
        'seaborn', 'fastapi', 'uvicorn'
    ]

    missing = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)

    if missing:
        print(f"❌ Missing required modules: {', '.join(missing)}")
        print("Please install them with: pip install -r requirements.txt")
        return False

    print("✅ All required dependencies found")
    return True


def download_nltk_data():
    """Download required NLTK data"""
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('brown', quiet=True)
        print("✅ NLTK data downloaded")
    except ImportError:
        print("⚠️ NLTK not available - TextBlob may not work properly")
    except Exception as e:
        print(f"⚠️ Could not download NLTK data: {e}")


def main():
    """Main startup function"""
    print("🎯 Starting Sentiment Analysis Tool...")
    print("=" * 50)

    # Setup
    setup_directories()

    if not check_dependencies():
        sys.exit(1)

    download_nltk_data()

    # Launch app
    print("\n🚀 Launching Streamlit app...")
    print("=" * 50)

    try:
        # Get port from command line args or default to 8501
        port = "8501"
        if len(sys.argv) > 1:
            port = sys.argv[1]

        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", port,
            "--server.address", "0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"❌ Error launching app: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()