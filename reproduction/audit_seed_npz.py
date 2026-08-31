"""Validate the local NPZ SEED DE export against DMMR's expected input."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_ROOT = Path("/media/NAS/nas_175/seojun/SEED_DE_MSMDA")


def load(root: Path, session: int, subject: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(root / f"session_{session}" / f"subject_{subject:02d}.npz") as archive:
        return archive["X"], archive["y"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--session", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--time-steps", type=int, default=30)
    args = parser.parse_args()
    manifest = json.loads((args.data_root / "manifest.json").read_text(encoding="utf-8"))
    failures, summaries = [], []
    for subject in range(1, 16):
        x, y = load(args.data_root, args.session, subject)
        entry = next(item for item in manifest["files"] if item["subject"] == subject and item["session"] == args.session)
        lengths = entry["trial_lengths"]
        ok = x.ndim == 2 and x.shape[1] == 310 and len(x) == len(y) == sum(lengths) and len(lengths) == 15
        start, labels, sequence_count = 0, [], 0
        for length in lengths:
            trial_y = np.unique(y[start:start + length])
            if len(trial_y) != 1:
                ok = False
            labels.append(int(trial_y[0]) if len(trial_y) == 1 else -1)
            sequence_count += max(0, length - args.time_steps + 1)
            start += length
        if not np.isfinite(x).all():
            ok = False
        if not ok:
            failures.append(subject)
        summaries.append({"subject": subject, "raw_windows": int(len(x)), "trial_lengths": lengths,
                          "trial_labels": labels, "dmmr_sequences": int(sequence_count), "finite": bool(np.isfinite(x).all())})
    result = {"data_root": str(args.data_root), "session": args.session, "feature_shape_expected": "(windows, 310) = 62 channels x 5 DE bands",
              "time_steps": args.time_steps, "subjects_valid": len(failures) == 0, "failed_subjects": failures,
              "subjects": summaries}
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit("NPZ audit failed")


if __name__ == "__main__":
    main()
