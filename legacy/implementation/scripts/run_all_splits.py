"""Run all task splits and generate comprehensive results.

Usage:
    python scripts/run_all_splits.py [--data-dir DIR] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_pipeline import run_single_split
from src.data_loader import load_dataset, dataset_summary
from src.evaluation import define_task_splits, save_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run all cross-task splits for gaze identification"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(PROJECT_ROOT.parent / "asreog_filter_order3_all_data"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "results"),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info("Loading dataset from %s ...", args.data_dir)
    t0 = time.time()
    dataset = load_dataset(args.data_dir, prefer_first_session=True)
    logger.info("Dataset loaded in %.1f seconds (%d subjects)", time.time() - t0, len(dataset))

    summary = dataset_summary(dataset)
    logger.info("Dataset: %d subjects", summary["n_subjects"])

    # Run all splits
    splits = define_task_splits()
    all_results = []

    for i, split in enumerate(splits):
        logger.info("\n%s", "=" * 70)
        logger.info("Running split %d/%d: %s", i + 1, len(splits), split["name"])
        logger.info("%s", "=" * 70)

        t0 = time.time()
        results = run_single_split(dataset, split, output_dir, args.random_state)
        elapsed = time.time() - t0
        logger.info("Split completed in %.1f seconds", elapsed)

        all_results.extend(results)

    # Save aggregated results
    save_results(all_results, str(output_dir))

    # Generate a markdown summary
    _generate_markdown_summary(all_results, output_dir / "REPORT.md")

    logger.info("\nAll splits completed. Results saved to %s", output_dir)


def _generate_markdown_summary(all_results: list, save_path: Path) -> None:
    """Generate a markdown report summarizing all results."""
    lines = [
        "# Cross-Task Gaze Identity: Results Report\n",
        "## Overview\n",
        "This report summarizes the results of cross-task person identification",
        "using eye-tracking gaze data from the EasyCog dataset.\n",
        "The goal is to determine whether individual gaze behavior remains",
        "recognizable when cognitive demands change.\n",
        "## Task Splits and Results\n",
    ]

    # Group by split
    splits_seen = {}
    for r in all_results:
        sn = r["split_name"]
        if sn not in splits_seen:
            splits_seen[sn] = {
                "info": r.get("split_info", {}),
                "models": [],
            }
        splits_seen[sn]["models"].append(r)

    for split_name, data in splits_seen.items():
        info = data["info"]
        lines.append(f"### {split_name}\n")
        if info.get("train_tasks"):
            lines.append(f"- **Train tasks**: {', '.join(info['train_tasks'])}")
        if info.get("test_tasks"):
            lines.append(f"- **Test tasks**: {', '.join(info['test_tasks'])}")
        if info.get("rationale"):
            lines.append(f"- **Rationale**: {info['rationale']}")
        lines.append("")

        lines.append("| Model | Accuracy | Balanced Acc | F1 Macro | F1 Weighted | Random Baseline |")
        lines.append("|-------|----------|--------------|----------|-------------|-----------------|")
        for r in data["models"]:
            m = r["metrics"]
            lines.append(
                f"| {r['model_name']} | {m['accuracy']:.4f} | {m['balanced_accuracy']:.4f} | "
                f"{m['f1_macro']:.4f} | {m['f1_weighted']:.4f} | {m['random_baseline']:.4f} |"
            )
        lines.append("")

    # Best overall result
    if all_results:
        best = max(all_results, key=lambda r: r["metrics"]["accuracy"])
        lines.append("## Best Overall Result\n")
        lines.append(f"- **Split**: {best['split_name']}")
        lines.append(f"- **Model**: {best['model_name']}")
        lines.append(f"- **Accuracy**: {best['metrics']['accuracy']:.4f}")
        lines.append(f"- **Balanced Accuracy**: {best['metrics']['balanced_accuracy']:.4f}")
        lines.append(f"- **F1 Macro**: {best['metrics']['f1_macro']:.4f}")
        lines.append(f"- **Random Baseline**: {best['metrics']['random_baseline']:.4f}")
        lines.append("")

    # Key observations
    lines.extend([
        "## Key Observations\n",
        "1. **Cross-task generalization**: Compare low-to-high vs same-level splits.",
        "2. **Model comparison**: Tree-based models (RF, XGBoost) vs linear models.",
        "3. **Cognitive load impact**: How does cognitive demand shift affect accuracy?",
        "4. **Scalability**: Performance with 69 classes (vs random baseline ~1.45%).\n",
        "## Figures\n",
        "- `results_summary.png`: Accuracy comparison across splits and models.",
        "- `cm_*.png`: Confusion matrices for best models per split.",
        "- `feat_imp_*.png`: Feature importance for tree-based models.\n",
    ])

    with open(save_path, "w") as f:
        f.write("\n".join(lines))
    logger.info("Markdown report saved to %s", save_path)


if __name__ == "__main__":
    main()
