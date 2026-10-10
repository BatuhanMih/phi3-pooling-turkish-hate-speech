"""Print the public aggregate result files as compact tables."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_json(name: str) -> dict:
    with (ROOT / "results" / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    development = load_json("development_5x2_summary.json")
    official = load_json("official_paired_summary.json")

    print("Development 5x2 Macro-F1")
    for condition, score in development["overall_mean_macro_f1"].items():
        print(f"  {condition}: {score:.6f}")

    print("\nOfficial paired evaluation")
    print(f"  P0: {official['mean_p0_macro_f1']:.6f}")
    print(f"  P5: {official['mean_p5_macro_f1']:.6f}")
    print(f"  P5-P0: {official['mean_paired_p5_minus_p0']:.6f}")


if __name__ == "__main__":
    main()

