"""Create an auditable cohort manifest from local video NPZ files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.data_contract import load_sessions, manifest, select_cohort


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT.parent / "asreog_filter_order3_all_data")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "baseline.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "cohort_manifest.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    sessions, load_exclusions = load_sessions(args.data_dir)
    cohort, selection_exclusions = select_cohort(sessions, **config["cohort"])
    output = manifest(cohort, [*load_exclusions, *selection_exclusions], config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Selected {output['n_subjects']} subjects / {output['n_sessions']} sessions")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
