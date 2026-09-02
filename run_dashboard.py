"""
Dashboard launcher script for PM2.5 Forecasting Dashboard.

This script provides an easy way to start the Streamlit dashboard
with proper environment setup and error checking.
"""

import sys
import subprocess
from pathlib import Path
import os

def check_requirements():
    """Check if required files exist."""
    print("🔍 Checking dashboard requirements...")
    
    required_files = [
        "dashboard/app.py",
        "dashboard/requirements.txt",
        "data/processed/pm25_features.csv",
        "reports/model_comparison.csv"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    print("✅ All required files found")
    return True

def install_dependencies():
    """Install dashboard dependencies."""
    print("📦 Installing dashboard dependencies...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", 
            "dashboard/requirements.txt"
        ], check=True, capture_output=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def start_dashboard():
    """Start the Streamlit dashboard."""
    print("🚀 Starting PM2.5 Forecasting Dashboard...")
    
    # Change to dashboard directory
    dashboard_dir = Path("dashboard")
    
    try:
        # Start Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py"
        ], cwd=dashboard_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start dashboard: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
        return True

def main():
    """Main launcher function."""
    print("🌫️ PM2.5 Forecasting Dashboard Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("dashboard").exists():
        print("❌ Please run this script from the project root directory")
        print("   (where the 'dashboard' folder is located)")
        sys.exit(1)
    
    # Check requirements
    if not check_requirements():
        print("\n💡 Please ensure all data processing steps are complete before running the dashboard.")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n💡 You can try installing dependencies manually:")
        print("   pip install -r dashboard/requirements.txt")
        sys.exit(1)
    
    # Start dashboard
    print("\n🌐 Dashboard will be available at: http://localhost:8501")
    print("📝 Press Ctrl+C to stop the dashboard")
    print("-" * 50)
    
    start_dashboard()

if __name__ == "__main__":
    main()
