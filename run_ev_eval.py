from __future__ import annotations

import argparse
import json
from pathlib import Path

from ev_langgraph import run


def main() -> None:
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run the UrbanEV LangGraph experiment")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project.parents[1] / "data" / "urbanev",
        help="Folder containing time.csv, volume.csv, occupancy.csv, and price.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=project / "outputs")
    args = parser.parse_args()
    result = run(args.data_dir, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

