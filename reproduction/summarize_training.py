"""Aggregate reproducible DMMR epoch logs without third-party plotting tools."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path, help="Directory containing subject_XX_artifacts folders")
    return parser.parse_args()


def load_records(output_dir: Path) -> tuple[list[dict], int]:
    records, subjects = [], 0
    for path in sorted(output_dir.glob("subject_*_artifacts/epoch_metrics.jsonl")):
        subjects += 1
        subject = path.parent.name.split("_")[1]
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            record["subject"] = int(subject)
            records.append(record)
    if not records:
        raise SystemExit(f"No epoch_metrics.jsonl files found below {output_dir}")
    return records, subjects


def aggregate(records: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for record in records:
        groups[(record["phase"], int(record["epoch"]))].append(record)
    rows = []
    for (phase, epoch), items in sorted(groups.items()):
        row = {"phase": phase, "epoch": epoch, "folds": len(items)}
        numeric_keys = sorted(set.intersection(*[
            {key for key, value in item.items() if isinstance(value, (int, float)) and key not in {"epoch", "subject"}}
            for item in items
        ]))
        for key in numeric_keys:
            values = [float(item[key]) for item in items]
            row[f"{key}_mean"] = statistics.mean(values)
            row[f"{key}_population_sd"] = statistics.pstdev(values)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def polyline(rows: list[dict], key: str, x0: int, y0: int, width: int, height: int) -> str:
    points = [(int(row["epoch"]), float(row[key])) for row in rows if key in row]
    if not points:
        return ""
    xmin, xmax = min(x for x, _ in points), max(x for x, _ in points)
    ymin, ymax = min(y for _, y in points), max(y for _, y in points)
    if ymin == ymax: ymax = ymin + 1
    coords = []
    for x, y in points:
        px = x0 + (x - xmin) / max(1, xmax - xmin) * width
        py = y0 + height - (y - ymin) / (ymax - ymin) * height
        coords.append(f"{px:.1f},{py:.1f}")
    return (f'<polyline fill="none" stroke="#2563eb" stroke-width="2" points="{" ".join(coords)}"/>'
            f'<text x="{x0}" y="{y0 - 8}" font-size="14">{key}: {ymin:.4f}–{ymax:.4f}</text>')


def write_svg(path: Path, rows: list[dict]) -> None:
    panels = [
        ("pretrain", "loss_mean"),
        ("pretrain", "reconstruction_loss_mean"),
        ("finetune", "loss_mean"),
        ("finetune", "validation_trial_accuracy_mean"),
        ("finetune", "train_minibatch_accuracy_mean"),
        ("finetune", "gradient_l2_norm_mean_mean"),
    ]
    elements = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="780" viewBox="0 0 1200 780">',
                '<rect width="100%" height="100%" fill="white"/>',
                '<text x="40" y="30" font-size="20" font-weight="bold">DMMR training curves (fold mean)</text>']
    for index, (phase, key) in enumerate(panels):
        x0, y0 = 50 + (index % 2) * 590, 70 + (index // 2) * 230
        subset = [row for row in rows if row["phase"] == phase]
        elements.append(f'<rect x="{x0}" y="{y0}" width="540" height="180" fill="none" stroke="#cbd5e1"/>')
        elements.append(polyline(subset, key, x0, y0, 540, 180))
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def write_summary(path: Path, output_dir: Path, records: list[dict], rows: list[dict], subjects: int) -> None:
    checkpoints = list(output_dir.glob("subject_*_artifacts/checkpoints/*.pt"))
    checkpoint_bytes = sum(item.stat().st_size for item in checkpoints)
    validation = [row for row in rows if row["phase"] == "finetune" and "validation_trial_accuracy_mean" in row]
    best = max(validation, key=lambda row: row["validation_trial_accuracy_mean"]) if validation else None
    lines = ["# DMMR training artifact summary", "", f"- Completed/logged folds: {subjects}",
             f"- Epoch records: {len(records)}", f"- Checkpoints: {len(checkpoints)} ({checkpoint_bytes / 1024**3:.2f} GiB)"]
    if best:
        lines.append(f"- Best fold-mean validation trial accuracy: epoch {best['epoch']} ({best['validation_trial_accuracy_mean']:.4%})")
    lines.extend(["", "## Files", "", "- `aggregate_epoch_metrics.csv`: epoch-wise fold means and population SD",
                  "- `training_curves.svg`: loss, accuracy, and gradient curves",
                  "- Per-fold metadata, exact command, source snapshot, hashes, metrics, and checkpoints are under each `subject_XX_artifacts/` directory.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args(); records, subjects = load_records(args.output_dir); rows = aggregate(records)
    write_csv(args.output_dir / "aggregate_epoch_metrics.csv", rows)
    write_svg(args.output_dir / "training_curves.svg", rows)
    write_summary(args.output_dir / "TRAINING_SUMMARY.md", args.output_dir, records, rows, subjects)
    print(f"Wrote aggregate metrics, summary, and curves to {args.output_dir}")


if __name__ == "__main__":
    main()
