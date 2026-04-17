from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quintic_sv.spx_deep_learning import (
    generate_spx_surrogate_dataset,
    save_spx_surrogate_model,
    train_spx_surrogate_model,
)
from quintic_sv.utils import ensure_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Numpy MLP that replaces SPX Monte Carlo.")
    parser.add_argument("--output", default=str(ROOT / "outputs" / "models" / "spx_surrogate_demo.npz"), help="Destination .npz file.")
    parser.add_argument("--number-of-experiments", type=int, default=48, help="Number of teacher experiments sampled for the dataset.")
    parser.add_argument("--strikes-per-experiment", type=int, default=14, help="Number of strikes priced for each teacher experiment.")
    parser.add_argument("--teacher-steps", type=int, default=64, help="Time steps used by the Monte Carlo teacher.")
    parser.add_argument("--teacher-base-paths", type=int, default=700, help="Base Monte Carlo paths before antithetic duplication.")
    parser.add_argument("--epochs", type=int, default=220, help="Training epochs for the dense network.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=3e-3, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used for dataset generation and training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directory(Path(args.output).parent)

    dataset = generate_spx_surrogate_dataset(
        number_of_experiments=args.number_of_experiments,
        strikes_per_experiment=args.strikes_per_experiment,
        teacher_steps=args.teacher_steps,
        teacher_base_paths=args.teacher_base_paths,
        seed=args.seed,
    )
    model, history = train_spx_surrogate_model(
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    destination = save_spx_surrogate_model(model, args.output)

    print(f"Dataset size: {dataset.features.shape[0]} samples")
    print(f"Final train loss: {history.train_loss[-1]:.6f}")
    print(f"Final validation loss: {history.validation_loss[-1]:.6f}")
    print(f"Model written to: {destination}")


if __name__ == "__main__":
    main()
