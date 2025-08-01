#!/usr/bin/env python3
"""
Simple runner script for LLM-Story-Board
"""

import os
import sys
from pathlib import Path

def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking requirements...")
    
    # Check if .env exists
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        print("📝 Please copy .env.example to .env and configure your API key")
        return False
    
    # Check if dataset directory exists
    dataset_dir = Path('dataset')
    if not dataset_dir.exists():
        print("📁 Creating dataset directory...")
        dataset_dir.mkdir(exist_ok=True)
        print("⚠️  Please place your image dataset JSON file in the dataset/ directory")
        print("   Expected filename: description-in-isolation.json")
    
    # Try to import required modules
    try:
        import flask
        import groq
        import flask_cors
        import flask_limiter
        print("✅ All dependencies installed")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("📦 Please run: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Main runner function"""
    print("🎬 LLM-Story-Board Starter")
    print("=" * 40)
    
    if not check_requirements():
        print("\n❌ Requirements check failed. Please fix the issues above.")
        sys.exit(1)
    
    print("\n🚀 Starting LLM-Story-Board...")
    print("📍 Server will be available at: http://localhost:5000")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 40)
    
    # Import and run the main application
    try:
        from llm_story3 import app, DATASET_LOADED, client
        
        if not DATASET_LOADED:
            print("❌ Dataset failed to load. Please check your dataset file.")
            sys.exit(1)
        
        if not client:
            print("❌ Groq client not initialized. Please check your API key in .env")
            sys.exit(1)
        
        app.run(
            host='0.0.0.0',
            port=int(os.getenv('PORT', 5000)),
            debug=os.getenv('DEBUG', 'False').lower() == 'true'
        )
        
    except ImportError:
        print("❌ Could not import llm-story3.py")
        print("📝 Make sure the file is named 'llm-story3.py' or update the import")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()