"""CLI entry points for PromptSearch."""

import subprocess
import sys
from pathlib import Path


def run_dashboard():
    """Launch the PromptSearch Streamlit dashboard."""
    # Get the path to the dashboard module
    dashboard_path = Path(__file__).parent / "dashboard.py"
    
    if not dashboard_path.exists():
        print(f"Error: Dashboard not found at {dashboard_path}")
        sys.exit(1)
    
    # Build streamlit command
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(dashboard_path),
        "--theme.base=dark",
        "--theme.primaryColor=#FF6B35",
        "--theme.backgroundColor=#0E1117",
        "--theme.secondaryBackgroundColor=#1E2128",
        "--theme.textColor=#FAFAFA",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ]
    
    # Add any additional arguments passed to the CLI
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])
    
    print("🔍 Starting PromptSearch Dashboard...")
    print(f"   Dashboard: {dashboard_path}")
    print()
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    except FileNotFoundError:
        print("Error: Streamlit not found. Install with: pip install streamlit")
        sys.exit(1)


if __name__ == "__main__":
    run_dashboard()

