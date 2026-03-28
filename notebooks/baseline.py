#!/usr/bin/env python
"""
Production baseline runner
uv run python notebooks/baseline.py --submit
"""

import typer
from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn
import pandas as pd
from pathlib import Path

from src.ensemble import create_production_ensemble, evaluate_ensemble
from arc.benchmark import load_dataset

app = typer.Typer()

@app.command()
def run(
    dataset: str = "arc-agi-2024-validation",
    max_tasks: int = 400,
    device: str = None,
    submit: bool = False
):
    """Production baseline evaluation"""
    
    print(f"🚀 Running ARC 2026 Baseline on [bold cyan]{dataset}[/bold cyan]")
    
    # Load model
    model = create_production_ensemble(device)
    
    # Evaluate
    tasks = load_dataset(dataset)[:max_tasks]
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=print.console
    ) as progress:
        task = progress.add_task("Evaluating...", total=len(tasks))
        
        results = {"correct": 0, "total": 0}
        for i, t in enumerate(tasks):
            pred = model.solve(t)
            true_out = utils.parse(t.test[0]["output"])
            pred_grid = utils.parse(pred)
            
            if np.array_equal(pred_grid, true_out):
                results["correct"] += 1
            
            results["total"] += 1
            progress.update(task, advance=1, description=f"Solved {i+1}/{len(tasks)}")
    
    accuracy = results["correct"] / results["total"]
    print(f"\n🏆 [bold green]FINAL SCORE: {accuracy:.1%} ({results['correct']}/{results['total']})[/bold green]")
    
    if submit:
        generate_submission(model, dataset)
        print("📤 [bold green]Kaggle submission ready![/bold green]")

if __name__ == "__main__":
    app()
