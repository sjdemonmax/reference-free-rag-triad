"""
start_dashboard.py - Helper script to launch Streamlit Dashboard
"""
import os
import subprocess
import sys

def main():
    dashboard_file = os.path.join(os.path.dirname(__file__), "final output 3.py")
    cmd = [
        "/opt/anaconda3/bin/streamlit",
        "run",
        dashboard_file,
        "--server.port=8501",
        "--server.headless=false",
        "--browser.gatherUsageStats=false"
    ]
    print(f"Launching Streamlit dashboard: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
