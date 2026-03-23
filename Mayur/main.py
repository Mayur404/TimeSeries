"""Simple entrypoint for rebuilding data and running the modeling pipeline."""

from __future__ import annotations

from dataset.data_preprocessing import run_preprocessing_pipeline
from model import run_training_pipeline


def main() -> None:
    print("=" * 70)
    print(" Time Series Assignment Project Runner")
    print("=" * 70)
    print("[1/2] Rebuilding the cleaned dataset from raw files...")
    run_preprocessing_pipeline()

    print("\n[2/2] Training and evaluating forecasting models...")
    leaderboard = run_training_pipeline()

    best_row = leaderboard.iloc[0]
    print("\nProject complete.")
    print(f"Best model: {best_row['Model']}")
    print(f"Test RMSE: {best_row['Test RMSE']:.3f}")
    print(f"Test MAE: {best_row['Test MAE']:.3f}")
    print(f"Test R2: {best_row['Test R2']:.3f}")


if __name__ == "__main__":
    main()
