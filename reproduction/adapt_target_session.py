"""Supervised target-session adaptation from completed LOSO checkpoints.

The target adaptation session is used with labels. Test sessions are disjoint
and evaluated only before adaptation and at the fixed final adaptation epoch.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import shlex
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "model.py").exists() else SCRIPT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from model import DMMRFineTuningModel, DMMRPreTrainingModel
from run_npz_loso import (DEFAULT_ROOT, atomic_torch_save, evaluate, gradient_l2_norm,
                          load_subject_sessions, parameter_l2_norm, rng_state, sha256)


DEFAULT_BASE = Path("/media/NAS/nas_175/seojun/DMMR/val_session3_cnn_pre50_ft100_cosine")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True, choices=range(1, 16))
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--adapt-session", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--test-sessions", type=int, nargs="+", default=[2, 3], choices=(1, 2, 3))
    parser.add_argument("--adapt-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--scheduler", choices=("none", "cosine"), default="cosine")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--adapt-scope", choices=("full", "head"), default="full")
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def plain_args(args: argparse.Namespace) -> dict:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def save_provenance(args: argparse.Namespace, artifact_dir: Path, base_checkpoint: Path) -> None:
    snapshot = artifact_dir / "source_snapshot"; snapshot.mkdir(parents=True, exist_ok=True)
    cuda_env = PROJECT_ROOT / "reproduction" / "launchers" / "dmmr_cuda_env.sh"
    if not cuda_env.exists():
        cuda_env = SCRIPT_DIR / "dmmr_cuda_env.sh"
    sources = [Path(__file__).resolve(), SCRIPT_DIR / "run_npz_loso.py",
               PROJECT_ROOT / "model.py", PROJECT_ROOT / "GradientReverseLayer.py", cuda_env]
    source_hashes = {}
    for source in sources:
        shutil.copyfile(source, snapshot / source.name); source_hashes[source.name] = sha256(source)
    data_files = [args.data_root / "manifest.json"] + [
        args.data_root / f"session_{session}" / f"subject_{args.subject:02d}.npz"
        for session in sorted(set([args.adapt_session, *args.test_sessions]))]
    metadata = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": plain_args(args),
        "base_checkpoint": str(base_checkpoint),
        "base_checkpoint_sha256": sha256(base_checkpoint),
        "source_sha256": source_hashes,
        "data_sha256": {str(path.relative_to(args.data_root)): sha256(path) for path in data_files},
        "python": sys.version,
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda,
                  "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "device": torch.cuda.get_device_name(torch.device(args.device))
                  if torch.cuda.is_available() and torch.device(args.device).type == "cuda" else None},
    }
    (artifact_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    reproduce = list(sys.argv[1:])
    reproduce[reproduce.index("--output-dir") + 1] = str(artifact_dir / "reproduced")
    command = [sys.executable, str(snapshot / Path(__file__).name), *reproduce]
    (artifact_dir / "reproduce_command.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nsource " + shlex.quote(str(snapshot / "dmmr_cuda_env.sh"))
        + "\n" + " ".join(shlex.quote(item) for item in command) + "\n", encoding="utf-8")


def write_metrics(artifact_dir: Path, history: list[dict]) -> None:
    jsonl = artifact_dir / "epoch_metrics.jsonl"
    jsonl.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in history), encoding="utf-8")
    fields = list(dict.fromkeys(key for row in history for key in row))
    with (artifact_dir / "epoch_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(history)


def infer_encoder(base_result: dict, saved: dict) -> str:
    """Resolve encoder type, including legacy runs that predate encoder metadata."""
    recorded = base_result.get("encoder")
    if recorded in {"cnn", "lstm"}:
        return recorded
    state = saved.get("pretrain", {})
    keys = tuple(state)
    if any("sharedEncoder.theta.lstm." in key for key in keys):
        return "lstm"
    if any("sharedEncoder.theta.input." in key for key in keys):
        return "cnn"
    raise ValueError("Could not infer encoder type from result metadata or checkpoint keys")


def main() -> None:
    args = parse_args(); args.test_sessions = list(dict.fromkeys(args.test_sessions))
    if args.adapt_session in args.test_sessions:
        raise ValueError("adapt-session must not appear in test-sessions")
    if args.adapt_epochs < 1 or args.checkpoint_every < 0:
        raise ValueError("adapt-epochs must be positive and checkpoint-every non-negative")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args.device); cuda = device.type == "cuda"

    base_checkpoint = args.base_dir / f"subject_{args.subject:02d}.pt"
    base_json = args.base_dir / f"subject_{args.subject:02d}.json"
    saved = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    base_result = json.loads(base_json.read_text(encoding="utf-8"))
    hyper = base_result["hyperparameters"]
    encoder = infer_encoder(base_result, saved)
    time_steps = int(hyper["time_steps"])
    pretrain = DMMRPreTrainingModel(cuda, number_of_source=14, number_of_category=3,
                                    batch_size=args.batch_size, time_steps=time_steps,
                                    encoder_type=encoder).to(device)
    pretrain.load_state_dict(saved["pretrain"])
    model = DMMRFineTuningModel(cuda, pretrain, number_of_source=14, number_of_category=3,
                                batch_size=args.batch_size, time_steps=time_steps).to(device)
    model.load_state_dict(saved["finetune"])
    for parameter in model.parameters(): parameter.requires_grad = False
    modules = [model.cls_fc] if args.adapt_scope == "head" else [model.attentionLayer, model.sharedEncoder, model.cls_fc]
    for module in modules:
        for parameter in module.parameters(): parameter.requires_grad = True
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.adapt_epochs, eta_min=args.lr * 0.01)
        if args.scheduler == "cosine" else None)

    ax, ay, adapt_trial_ids = load_subject_sessions(args.data_root, [args.adapt_session], args.subject, time_steps)
    adapt_train_loader = DataLoader(TensorDataset(ax, ay[:, None]), args.batch_size, shuffle=True,
                                    drop_last=False, num_workers=0)
    adapt_eval_loader = DataLoader(TensorDataset(ax, ay[:, None]), args.batch_size, shuffle=False,
                                   drop_last=False, num_workers=0)
    tx, ty, test_trial_ids = load_subject_sessions(args.data_root, args.test_sessions, args.subject, time_steps)
    test_loader = DataLoader(TensorDataset(tx, ty[:, None]), args.batch_size, shuffle=False,
                             drop_last=False, num_workers=0)
    baseline = evaluate(model, test_loader, test_trial_ids, device)
    baseline_by_session = {}
    for session in args.test_sessions:
        sx, sy, ids = load_subject_sessions(args.data_root, [session], args.subject, time_steps)
        loader = DataLoader(TensorDataset(sx, sy[:, None]), args.batch_size, shuffle=False, drop_last=False)
        baseline_by_session[str(session)] = evaluate(model, loader, ids, device)

    artifact_dir = args.output_dir / f"subject_{args.subject:02d}_artifacts"
    checkpoints = artifact_dir / "checkpoints"; checkpoints.mkdir(parents=True, exist_ok=True)
    save_provenance(args, artifact_dir, base_checkpoint)
    history = []
    for epoch in range(1, args.adapt_epochs + 1):
        started = time.perf_counter(); model.train(); losses = []; predictions = []; truths = []; gradients = []
        for x, y in adapt_train_loader:
            output, _, loss = model(x.to(device), y.to(device))
            optimizer.zero_grad(set_to_none=True); loss.backward()
            gradients.append(float(torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip))
                             if args.grad_clip > 0 else gradient_l2_norm(model))
            optimizer.step(); losses.append(float(loss.detach()))
            predictions.extend(output.detach().argmax(1).cpu().tolist()); truths.extend(y.squeeze(1).tolist())
        adapt_metrics = evaluate(model, adapt_eval_loader, adapt_trial_ids, device)
        record = {"epoch": epoch, "loss": float(np.mean(losses)), "lr": float(optimizer.param_groups[0]["lr"]),
                  "gradient_l2_norm_mean": float(np.mean(gradients)), "parameter_l2_norm": parameter_l2_norm(model),
                  "train_minibatch_accuracy": float(np.mean(np.asarray(predictions) == truths)),
                  "train_minibatch_macro_f1": float(f1_score(truths, predictions, average="macro", zero_division=0)),
                  **{f"adapt_{key}": value for key, value in adapt_metrics.items()},
                  "duration_seconds": time.perf_counter() - started}
        history.append(record); write_metrics(artifact_dir, history)
        if scheduler is not None: scheduler.step()
        if args.checkpoint_every and (epoch % args.checkpoint_every == 0 or epoch == args.adapt_epochs):
            atomic_torch_save({"schema_version": 1, "epoch": epoch, "model": model.state_dict(),
                               "optimizer": optimizer.state_dict(),
                               "scheduler": scheduler.state_dict() if scheduler else None,
                               "rng_state": rng_state(), "args": plain_args(args), "history": history},
                              checkpoints / f"adapt_epoch_{epoch:04d}.pt")
        print(f"subject={args.subject:02d} adapt={epoch}/{args.adapt_epochs} loss={record['loss']:.4f} "
              f"adapt_trial={adapt_metrics['trial_accuracy']:.4f}", flush=True)

    final_test = evaluate(model, test_loader, test_trial_ids, device)
    final_by_session = {}
    for session in args.test_sessions:
        sx, sy, ids = load_subject_sessions(args.data_root, [session], args.subject, time_steps)
        loader = DataLoader(TensorDataset(sx, sy[:, None]), args.batch_size, shuffle=False, drop_last=False)
        final_by_session[str(session)] = evaluate(model, loader, ids, device)
    result = {"schema_version": 1, "subject": args.subject, "encoder": encoder,
              "protocol": f"Supervised target session {args.adapt_session} adaptation; fixed-final evaluation on target sessions {args.test_sessions}.",
              "base_checkpoint": str(base_checkpoint), "base_selection": base_result.get("selection_metric"),
              "arguments": plain_args(args), "baseline_test": baseline, "baseline_by_session": baseline_by_session,
              "final_test": final_test, "final_by_session": final_by_session, "history": history}
    output = args.output_dir / f"subject_{args.subject:02d}.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    atomic_torch_save({"model": model.state_dict(), "result": result}, output.with_suffix(".pt"))
    print(f"saved {output} baseline_trial={baseline['trial_accuracy']:.4f} final_trial={final_test['trial_accuracy']:.4f}")


if __name__ == "__main__":
    main()
