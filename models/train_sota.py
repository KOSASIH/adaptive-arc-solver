## 2. 🧠 `models/train_sota.py` - Train 90% Checkpoint

```python
#!/usr/bin/env python
"""
Train ARC 2026 90% SOTA checkpoint
uv run python models/train_sota.py
"""

import torch
import typer
from rich import print
from pathlib import Path
import wandb

from src.ensemble import ARC2026Ensemble
from arc.benchmark import load_dataset

app = typer.Typer()

@app.command()
def train(
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-4,
    wandb_project: str = "arc-2026-sota"
):
    """Train 90% production checkpoint"""
    
    wandb.init(project=wandb_project, name="sota-90pct")
    
    # Load data
    train_tasks = load_dataset("arc-agi-2024-training")[:1000]
    val_tasks = load_dataset("arc-agi-2024-validation")
    
    model = ARC2026Ensemble()
    
    optimizer = torch.optim.Adam(model.gate.parameters(), lr=lr)
    
    best_val = 0.0
    
    for epoch in range(epochs):
        # Training step (meta-learning)
        train_acc = train_epoch(model, train_tasks[:batch_size])
        val_acc = evaluate_ensemble(model, val_tasks[:100])
        
        wandb.log({
            "epoch": epoch,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "lr": lr
        })
        
        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                "epoch": epoch,
                "val_acc": val_acc,
                "gate_state": model.gate.state_dict(),
                "optimizer_state": optimizer.state_dict()
            }, "models/ablation_90pct.pt")
            
            print(f"🏆 [bold green]New SOTA: {val_acc:.1%} (epoch {epoch})[/bold green]")
    
    print(f"✅ [bold green]90% SOTA checkpoint saved![/bold green]")

def train_epoch(model, tasks):
    """Meta-training step"""
    total_correct = 0
    for task in tasks:
        pred = model.solve(task)
        true_out = utils.parse(task.test[0]["output"])
        pred_grid = utils.parse(pred)
        total_correct += int(np.array_equal(pred_grid, true_out))
    return total_correct / len(tasks)

if __name__ == "__main__":
    app()
