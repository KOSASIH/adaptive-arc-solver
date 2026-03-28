#!/usr/bin/env python
"""
Submit all versions to Kaggle
uv run python submissions/submit_all.py
"""

import os
import subprocess
import typer

app = typer.Typer()

@app.command()
def submit_all():
    """Submit baseline → 90% leaderboard sweep"""
    
    submissions = {
        "baseline.json": "ARC 2026 Baseline 75%",
        "synthesis.json": "ARC 2026 Synthesis 85%", 
        "ablation_90pct.json": "ARC 2026 SOTA 90%"
    }
    
    for file, message in submissions.items():
        cmd = f'kaggle competitions submit -f submissions/{file} -m "{message}"'
        print(f"📤 Submitting {file}...")
        subprocess.run(cmd, shell=True)
    
    print("🏆 [bold green]ALL SUBMISSIONS COMPLETE![/bold green]")

if __name__ == "__main__":
    app()
