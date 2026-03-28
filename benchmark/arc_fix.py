"""
Fix ARC dataset paths
"""

from pathlib import Path
import json

def fix_arc_datasets():
    """Ensure datasets exist"""
    
    datasets = {
        "arc-agi-2024-validation": "ARC-AGI/data/validation",
        "arc-agi-2024-training": "ARC-AGI/data/training",
        "arc-agi-2024-test": "ARC-AGI/data/test"
    }
    
    for name, path in datasets.items():
        dataset_path = Path(path) / f"{name}.jsonl"
        if not dataset_path.exists():
            print(f"❌ Missing: {dataset_path}")
            print("🔧 Run: cd ARC-AGI && python scripts/download_arc_datasets.py")
        else:
            print(f"✅ Found: {dataset_path}")

if __name__ == "__main__":
    fix_arc_datasets()
