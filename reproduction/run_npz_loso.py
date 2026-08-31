"""Official DMMR architecture on the workspace's SEED DE NPZ export.

The default protocol is all-session subject-independent LOSO: sessions 1, 2,
and 3 from 14 subjects are used for training, while all three sessions from the
held-out subject are used only for evaluation.  The original model.py is
retained; this runner replaces its MATLAB loader and reports final-vs-best-test
metrics explicitly.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shlex
import shutil
import subprocess
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
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from model import DMMRPreTrainingModel, DMMRFineTuningModel

DEFAULT_ROOT = Path("/media/NAS/nas_175/seojun/SEED_DE_MSMDA")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--subject", type=int, required=True, choices=range(1, 16))
    parser.add_argument("--sessions", type=int, nargs="+", default=[1, 2, 3], choices=(1, 2, 3))
    parser.add_argument("--validation-session", type=int, choices=(1, 2, 3), default=None,
                        help="Reserve this session from every source subject for model selection.")
    parser.add_argument("--validation-trials-per-class", type=int, default=0,
                        help="Reserve this many whole trials per class and source subject, balanced across sessions.")
    parser.add_argument("--time-steps", type=int, default=30)
    parser.add_argument("--encoder", choices=("lstm", "cnn"), default="lstm")
    parser.add_argument("--pretrain-epochs", type=int, default=200)
    parser.add_argument("--pretrained-dir", type=Path, default=None,
                        help="Load this directory's subject_XX.pt stage-1 weights and skip pretraining.")
    parser.add_argument("--finetune-epochs", type=int, default=200)
    parser.add_argument("--iteration", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--scheduler", choices=("none", "cosine"), default="none")
    parser.add_argument("--grad-clip", type=float, default=0.0,
                        help="Maximum gradient norm; zero disables clipping.")
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint-every", type=int, default=5,
                        help="Save model weights every N epochs; zero disables periodic checkpoints.")
    parser.add_argument("--evaluate-test-every", type=int, default=0,
                        help="Diagnostic only: evaluate the held-out test subject every N finetune epochs; zero disables it.")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True,
                        help="Request deterministic PyTorch/CUDA algorithms (default: enabled).")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_npz_all_sessions_loso_200ep"))
    return parser.parse_args()


def jsonable_args(args: argparse.Namespace) -> dict:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*arguments: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *arguments], check=True, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parameter_l2_norm(model: torch.nn.Module) -> float:
    total = sum(float(parameter.detach().float().pow(2).sum()) for parameter in model.parameters())
    return math.sqrt(total)


def gradient_l2_norm(model: torch.nn.Module) -> float:
    total = sum(float(parameter.grad.detach().float().pow(2).sum())
                for parameter in model.parameters() if parameter.grad is not None)
    return math.sqrt(total)


def rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def atomic_torch_save(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def append_metrics(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_metrics_csv(path: Path, records: list[dict]) -> None:
    fields = list(dict.fromkeys(key for record in records for key in record))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def save_checkpoint(path: Path, phase: str, epoch: int, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer, scheduler, args: argparse.Namespace,
                    history: list[dict]) -> None:
    atomic_torch_save({
        "schema_version": 1,
        "phase": phase,
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "rng_state": rng_state(),
        "args": jsonable_args(args),
        "history": history,
    }, path)


def write_run_provenance(args: argparse.Namespace, run_dir: Path) -> dict:
    cuda_env = PROJECT_ROOT / "reproduction" / "launchers" / "dmmr_cuda_env.sh"
    if not cuda_env.exists():
        cuda_env = SCRIPT_DIR / "dmmr_cuda_env.sh"
    source_files = [Path(__file__).resolve(), PROJECT_ROOT / "model.py",
                    PROJECT_ROOT / "GradientReverseLayer.py", cuda_env]
    snapshot_dir = run_dir / "source_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    source_hashes = {}
    for source in source_files:
        shutil.copyfile(source, snapshot_dir / source.name)
        source_hashes[source.name] = sha256(source)

    data_files = [args.data_root / "manifest.json"] + [
        args.data_root / f"session_{session}" / f"subject_{subject:02d}.npz"
        for session in args.sessions for subject in range(1, 16)
    ]
    data_hashes = {str(path.relative_to(args.data_root)): sha256(path) for path in data_files}
    packages = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
                if dist.metadata.get("Name")}
    metadata = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "working_directory": str(Path.cwd()),
        "arguments": jsonable_args(args),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_status": git_value("status", "--short"),
        "source_sha256": source_hashes,
        "data_sha256": data_hashes,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": dict(sorted(packages.items())),
        "torch": {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_name": torch.cuda.get_device_name(torch.device(args.device))
                           if torch.cuda.is_available() and torch.device(args.device).type == "cuda" else None,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    reproduce_args = list(sys.argv[1:])
    if "--output-dir" in reproduce_args:
        reproduce_args[reproduce_args.index("--output-dir") + 1] = str(run_dir / "reproduced")
    else:
        reproduce_args.extend(["--output-dir", str(run_dir / "reproduced")])
    reproduce_command = [sys.executable, str(snapshot_dir / Path(__file__).name), *reproduce_args]
    command = " ".join(shlex.quote(item) for item in reproduce_command)
    (run_dir / "reproduce_command.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nsource "
        + shlex.quote(str(snapshot_dir / "dmmr_cuda_env.sh")) + "\n" + command + "\n",
        encoding="utf-8")
    (run_dir / "packages.txt").write_text(
        "".join(f"{name}=={version}\n" for name, version in metadata["packages"].items()),
        encoding="utf-8")
    return metadata


def raw(root: Path, session: int, subject: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    with np.load(root / f"session_{session}" / f"subject_{subject:02d}.npz") as archive:
        x, y = archive["X"].astype(np.float32), archive["y"].astype(np.int64)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    entry = next(item for item in manifest["files"] if item["subject"] == subject and item["session"] == session)
    return x, y, entry["trial_lengths"]


def make_sequences(x: np.ndarray, y: np.ndarray, lengths: list[int], time_steps: int,
                   normalize: bool = True) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    parts, labels, trial_ids = [], [], []
    start = 0
    for trial, length in enumerate(lengths):
        trial_x, trial_y = x[start:start + length], y[start:start + length]
        if len(np.unique(trial_y)) != 1:
            raise ValueError(f"trial {trial} has inconsistent labels")
        windows = len(trial_x) - time_steps + 1
        if windows <= 0:
            raise ValueError(f"trial {trial} shorter than {time_steps}")
        sequence = np.stack([trial_x[i:i + time_steps] for i in range(windows)])
        parts.append(sequence); labels.append(np.full(windows, trial_y[0], dtype=np.int64)); trial_ids.append(np.full(windows, trial, dtype=np.int64))
        start += length
    features = torch.from_numpy(np.concatenate(parts)).float()
    if normalize:
        # Match original preprocess.normalize(features, select_dim=0): independent
        # min/max for every timestep and DE feature within a subject.
        low, high = features.amin(dim=0, keepdim=True), features.amax(dim=0, keepdim=True)
        features = (features - low) / (high - low).clamp_min(1e-6)
    return features, torch.from_numpy(np.concatenate(labels)), np.concatenate(trial_ids)


def load_subject_sessions(root: Path, sessions: list[int], subject: int, time_steps: int) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Load sessions independently, then concatenate with globally unique trial IDs.

    Normalization remains session-local, matching the original per-file DMMR
    preprocessing and avoiding scale leakage between recording sessions.
    """
    features, labels, trial_ids = [], [], []
    trial_offset = 0
    for session in sessions:
        sx, sy, session_trials = make_sequences(*raw(root, session, subject), time_steps)
        features.append(sx)
        labels.append(sy)
        trial_ids.append(session_trials + trial_offset)
        trial_offset += int(session_trials.max()) + 1
    return torch.cat(features), torch.cat(labels), np.concatenate(trial_ids)


def load_subject_pool(root: Path, sessions: list[int], subjects: list[int], time_steps: int) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Concatenate subjects while keeping trial IDs unique across the pool."""
    features, labels, trial_ids = [], [], []
    trial_offset = 0
    for subject in subjects:
        sx, sy, subject_trials = load_subject_sessions(root, sessions, subject, time_steps)
        features.append(sx)
        labels.append(sy)
        trial_ids.append(subject_trials + trial_offset)
        trial_offset += int(subject_trials.max()) + 1
    return torch.cat(features), torch.cat(labels), np.concatenate(trial_ids)


def load_subject_trial_split(root: Path, sessions: list[int], subject: int, time_steps: int,
                             trials_per_class: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, np.ndarray,
                                                                       torch.Tensor, torch.Tensor, np.ndarray,
                                                                       dict[str, list[int]]]:
    """Create a deterministic class/session-balanced whole-trial split.

    Min/max statistics are fitted only on training trials in each session and
    then applied to validation trials from that session.
    """
    if trials_per_class % len(sessions) != 0:
        raise ValueError("validation-trials-per-class must be divisible by the number of sessions")
    per_session_class = trials_per_class // len(sessions)
    train_xs, train_ys, train_ids = [], [], []
    val_xs, val_ys, val_ids = [], [], []
    train_offset = val_offset = 0
    selected: dict[str, list[int]] = {}
    for session in sessions:
        features, labels, trial_ids = make_sequences(*raw(root, session, subject), time_steps, normalize=False)
        trial_labels = {int(trial): int(labels[np.flatnonzero(trial_ids == trial)[0]]) for trial in np.unique(trial_ids)}
        validation_trials: set[int] = set()
        for cls in range(3):
            candidates = [trial for trial, label in trial_labels.items() if label == cls]
            rng = random.Random(seed + subject * 1000 + session * 100 + cls)
            if len(candidates) < per_session_class:
                raise ValueError(f"subject {subject} session {session} class {cls} has too few trials")
            validation_trials.update(rng.sample(candidates, per_session_class))
        selected[f"session_{session}"] = sorted(validation_trials)
        val_mask = np.isin(trial_ids, list(validation_trials))
        train_mask = ~val_mask
        train_features, val_features = features[train_mask], features[val_mask]
        low, high = train_features.amin(dim=0, keepdim=True), train_features.amax(dim=0, keepdim=True)
        scale = (high - low).clamp_min(1e-6)
        train_features = (train_features - low) / scale
        val_features = (val_features - low) / scale
        session_train_ids, session_val_ids = trial_ids[train_mask], trial_ids[val_mask]
        train_xs.append(train_features); train_ys.append(labels[train_mask]); train_ids.append(session_train_ids + train_offset)
        val_xs.append(val_features); val_ys.append(labels[val_mask]); val_ids.append(session_val_ids + val_offset)
        train_offset += int(session_train_ids.max()) + 1
        val_offset += int(session_val_ids.max()) + 1
    return (torch.cat(train_xs), torch.cat(train_ys), np.concatenate(train_ids),
            torch.cat(val_xs), torch.cat(val_ys), np.concatenate(val_ids), selected)


def paired_correspondence(batches: list[tuple[torch.Tensor, torch.Tensor]], labels: torch.Tensor, rng: random.Random) -> torch.Tensor:
    """Pick one same-emotion sequence from every source for each current item."""
    banks: list[dict[int, list[torch.Tensor]]] = []
    for data, target in batches:
        bank: dict[int, list[torch.Tensor]] = {0: [], 1: [], 2: []}
        for item, cls in zip(data, target.squeeze(1).tolist()):
            bank[int(cls)].append(item)
        banks.append(bank)
    if any(not bank[int(cls)] for bank in banks for cls in labels.squeeze(1).tolist()):
        raise RuntimeError("A source mini-batch lacks a class; reduce batch size or resample.")
    selected = []
    for bank in banks:
        selected.extend(rng.choice(bank[int(cls)]) for cls in labels.squeeze(1).tolist())
    return torch.stack(selected)


@torch.no_grad()
def evaluate(model: DMMRFineTuningModel, loader: DataLoader, trial_ids: np.ndarray, device: torch.device) -> dict[str, float]:
    model.eval(); probabilities, truth = [], []
    for x, y in loader:
        x = x.to(device)
        attended = model.attentionLayer(x, x.shape[0], model.time_steps)
        final, _, _ = model.sharedEncoder(attended)
        probabilities.append(model.cls_fc(final).softmax(dim=1).cpu())
        truth.append(y.squeeze(1))
    probability, target = torch.cat(probabilities).numpy(), torch.cat(truth).numpy()
    prediction = probability.argmax(1)
    trial_prediction, trial_truth = [], []
    for trial in np.unique(trial_ids):
        index = np.flatnonzero(trial_ids == trial)
        trial_prediction.append(probability[index].mean(0).argmax())
        trial_truth.append(target[index][0])
    return {"sample_accuracy": float((prediction == target).mean()), "sample_macro_f1": float(f1_score(target, prediction, average="macro", zero_division=0)),
            "trial_accuracy": float(np.mean(np.asarray(trial_prediction) == trial_truth)), "trial_macro_f1": float(f1_score(trial_truth, trial_prediction, average="macro", zero_division=0)),
            "number_of_trials": int(len(trial_truth))}


def cycle_next(iterators: list, loaders: list[DataLoader], index: int):
    try:
        return next(iterators[index])
    except StopIteration:
        iterators[index] = iter(loaders[index])
        return next(iterators[index])


def main() -> None:
    args = parse_args()
    args.sessions = list(dict.fromkeys(args.sessions))
    if args.validation_session is not None and args.validation_trials_per_class:
        raise ValueError("choose validation-session or validation-trials-per-class, not both")
    if args.validation_session is not None and args.validation_session not in args.sessions:
        raise ValueError("validation-session must be included in sessions")
    train_sessions = ([session for session in args.sessions if session != args.validation_session]
                      if args.validation_session is not None else args.sessions)
    if not train_sessions:
        raise ValueError("at least one training session must remain")
    if args.checkpoint_every < 0 or args.evaluate_test_every < 0:
        raise ValueError("checkpoint-every and evaluate-test-every must be zero or positive")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = args.deterministic
    torch.use_deterministic_algorithms(args.deterministic, warn_only=True)
    rng = random.Random(args.seed + args.subject)
    device = torch.device(args.device); cuda = device.type == "cuda"
    run_dir = args.output_dir / f"subject_{args.subject:02d}_artifacts"
    checkpoint_dir = run_dir / "checkpoints"
    metrics_file = run_dir / "epoch_metrics.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)
    if metrics_file.exists():
        metrics_file.unlink()
    provenance = write_run_provenance(args, run_dir)
    source_subjects = [source for source in range(1, 16) if source != args.subject]
    source_loaders, split_manifest = [], {}
    pooled_val_x, pooled_val_y, pooled_val_ids = [], [], []
    pooled_val_offset = 0
    for source in source_subjects:
        if args.validation_trials_per_class:
            sx, sy, _, vx, vy, source_val_ids, selected = load_subject_trial_split(
                args.data_root, args.sessions, source, args.time_steps,
                args.validation_trials_per_class, args.seed)
            pooled_val_x.append(vx); pooled_val_y.append(vy)
            pooled_val_ids.append(source_val_ids + pooled_val_offset)
            pooled_val_offset += int(source_val_ids.max()) + 1
            split_manifest[str(source)] = selected
        else:
            sx, sy, _ = load_subject_sessions(args.data_root, train_sessions, source, args.time_steps)
        source_loaders.append(DataLoader(TensorDataset(sx, sy[:, None]), args.batch_size, shuffle=True, drop_last=True, num_workers=0))
    val_loader = val_trial_ids = None
    if args.validation_trials_per_class:
        vx, vy, val_trial_ids = torch.cat(pooled_val_x), torch.cat(pooled_val_y), np.concatenate(pooled_val_ids)
        val_loader = DataLoader(TensorDataset(vx, vy[:, None]), args.batch_size, shuffle=False, drop_last=False, num_workers=0)
    elif args.validation_session is not None:
        vx, vy, val_trial_ids = load_subject_pool(args.data_root, [args.validation_session], source_subjects, args.time_steps)
        val_loader = DataLoader(TensorDataset(vx, vy[:, None]), args.batch_size, shuffle=False, drop_last=False, num_workers=0)
    tx, ty, test_trial_ids = load_subject_sessions(args.data_root, args.sessions, args.subject, args.time_steps)
    test_loader = DataLoader(TensorDataset(tx, ty[:, None]), args.batch_size, shuffle=False, drop_last=False, num_workers=0)
    if any(len(loader) == 0 for loader in source_loaders):
        raise ValueError("batch-size exceeds number of trial-contained sequences")

    pretrain = DMMRPreTrainingModel(cuda, number_of_source=14, number_of_category=3,
                                    batch_size=args.batch_size, time_steps=args.time_steps,
                                    encoder_type=args.encoder).to(device)
    stage1_source = None
    if args.pretrained_dir is not None:
        stage1_source = args.pretrained_dir / f"subject_{args.subject:02d}.pt"
        saved_stage1 = torch.load(stage1_source, map_location="cpu")
        saved_result = saved_stage1.get("result", {})
        if saved_result.get("encoder", args.encoder) != args.encoder:
            raise ValueError(f"stage-1 encoder mismatch in {stage1_source}")
        pretrain.load_state_dict(saved_stage1["pretrain"])
        pretrain_history = saved_result.get("pretrain_history", [])
        print(f"subject={args.subject:02d} loaded_stage1={stage1_source}", flush=True)
    else:
        pre_opt = torch.optim.Adam(pretrain.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        pre_scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
            pre_opt, T_max=args.pretrain_epochs, eta_min=args.lr * 0.01)
            if args.scheduler == "cosine" else None)
        source_iterators = [iter(loader) for loader in source_loaders]
        pretrain_history = []
        for epoch in range(1, args.pretrain_epochs + 1):
            epoch_started = time.perf_counter()
            pretrain.train(); losses = []; gradient_norms = []
            for step in range(1, args.iteration + 1):
                progress = (step + (epoch - 1) * args.iteration) / (args.pretrain_epochs * args.iteration)
                grl = 2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0
                batches = [cycle_next(source_iterators, source_loaders, index) for index in range(14)]
                for domain, (x, y) in enumerate(batches):
                    correspondence = paired_correspondence(batches, y, rng).to(device)
                    subject_id = torch.full((args.batch_size,), domain, dtype=torch.long, device=device)
                    rec, domain_loss = pretrain(x.to(device), correspondence, subject_id, grl)
                    loss = rec + args.beta * domain_loss
                    pre_opt.zero_grad(set_to_none=True); loss.backward()
                    if args.grad_clip > 0:
                        grad_norm = float(torch.nn.utils.clip_grad_norm_(pretrain.parameters(), args.grad_clip))
                    else:
                        grad_norm = gradient_l2_norm(pretrain)
                    gradient_norms.append(grad_norm)
                    pre_opt.step(); losses.append((float(loss.detach()), float(rec.detach()), float(domain_loss.detach())))
            record = {"phase": "pretrain", "epoch": epoch,
                      "loss": float(np.mean([x[0] for x in losses])),
                      "reconstruction_loss": float(np.mean([x[1] for x in losses])),
                      "subject_loss": float(np.mean([x[2] for x in losses])),
                      "lr": float(pre_opt.param_groups[0]["lr"]),
                      "gradient_l2_norm_mean": float(np.mean(gradient_norms)),
                      "parameter_l2_norm": parameter_l2_norm(pretrain),
                      "grl_coefficient_final": float(grl),
                      "duration_seconds": time.perf_counter() - epoch_started}
            pretrain_history.append(record); print(f"subject={args.subject:02d} pretrain={epoch}/{args.pretrain_epochs} loss={record['loss']:.4f}", flush=True)
            append_metrics(metrics_file, record)
            if pre_scheduler is not None:
                pre_scheduler.step()
            if args.checkpoint_every and (epoch % args.checkpoint_every == 0 or epoch == args.pretrain_epochs):
                save_checkpoint(checkpoint_dir / f"pretrain_epoch_{epoch:04d}.pt", "pretrain", epoch,
                                pretrain, pre_opt, pre_scheduler, args, pretrain_history)

    fine = DMMRFineTuningModel(cuda, pretrain, number_of_source=14, number_of_category=3, batch_size=args.batch_size, time_steps=args.time_steps).to(device)
    fine_opt = torch.optim.Adam(fine.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    fine_scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
        fine_opt, T_max=args.finetune_epochs, eta_min=args.lr * 0.01)
        if args.scheduler == "cosine" else None)
    fine_iterators = [iter(loader) for loader in source_loaders]
    history, best_validation, best_state = [], None, None
    for epoch in range(1, args.finetune_epochs + 1):
        epoch_started = time.perf_counter()
        fine.train(); losses = []; gradient_norms = []; train_prediction = []; train_truth = []
        for _ in range(args.iteration):
            for domain in range(14):
                x, y = cycle_next(fine_iterators, source_loaders, domain)
                prediction, _, loss = fine(x.to(device), y.to(device))
                fine_opt.zero_grad(set_to_none=True); loss.backward()
                if args.grad_clip > 0:
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(fine.parameters(), args.grad_clip))
                else:
                    grad_norm = gradient_l2_norm(fine)
                gradient_norms.append(grad_norm)
                fine_opt.step(); losses.append(float(loss.detach()))
                train_prediction.extend(prediction.detach().argmax(dim=1).cpu().tolist())
                train_truth.extend(y.squeeze(1).tolist())
        base_record = {"phase": "finetune", "epoch": epoch,
                       "loss": float(np.mean(losses)),
                       "lr": float(fine_opt.param_groups[0]["lr"]),
                       "gradient_l2_norm_mean": float(np.mean(gradient_norms)),
                       "parameter_l2_norm": parameter_l2_norm(fine),
                       "train_minibatch_accuracy": float(np.mean(np.asarray(train_prediction) == train_truth)),
                       "train_minibatch_macro_f1": float(f1_score(train_truth, train_prediction, average="macro", zero_division=0))}
        if val_loader is not None:
            metrics = evaluate(fine, val_loader, val_trial_ids, device)
            prefix = "validation"
            record = {**base_record, **{f"{prefix}_{key}": value for key, value in metrics.items()}}
            score = (metrics["trial_accuracy"], metrics["trial_macro_f1"], metrics["sample_accuracy"])
            if best_validation is None or score > best_validation["selection_score"]:
                best_validation = {**record, "selection_score": score}
                best_state = copy.deepcopy(fine.state_dict())
                atomic_torch_save({"schema_version": 1, "phase": "finetune", "epoch": epoch,
                                   "selection_score": score, "model": best_state,
                                   "args": jsonable_args(args)}, checkpoint_dir / "finetune_best_validation.pt")
            print(f"subject={args.subject:02d} finetune={epoch}/{args.finetune_epochs} loss={record['loss']:.4f} {prefix}_sample={metrics['sample_accuracy']:.4f} {prefix}_trial={metrics['trial_accuracy']:.4f}", flush=True)
        else:
            record = base_record
            if args.evaluate_test_every and (epoch % args.evaluate_test_every == 0 or epoch == args.finetune_epochs):
                diagnostic_test = evaluate(fine, test_loader, test_trial_ids, device)
                record.update({f"diagnostic_test_{key}": value for key, value in diagnostic_test.items()})
            print(f"subject={args.subject:02d} finetune={epoch}/{args.finetune_epochs} loss={record['loss']:.4f}", flush=True)
        record["duration_seconds"] = time.perf_counter() - epoch_started
        history.append(record)
        append_metrics(metrics_file, record)
        if fine_scheduler is not None:
            fine_scheduler.step()
        if args.checkpoint_every and (epoch % args.checkpoint_every == 0 or epoch == args.finetune_epochs):
            save_checkpoint(checkpoint_dir / f"finetune_epoch_{epoch:04d}.pt", "finetune", epoch,
                            fine, fine_opt, fine_scheduler, args, history)
    final_test = evaluate(fine, test_loader, test_trial_ids, device)
    if best_state is None:
        best_state = copy.deepcopy(fine.state_dict())
        selected_test = final_test
    else:
        fine.load_state_dict(best_state)
        selected_test = evaluate(fine, test_loader, test_trial_ids, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = (f"Trial-validation LOSO: {args.validation_trials_per_class} whole trials per class from every source subject validate; "
                f"remaining source trials train; held-out subject sessions {args.sessions} test after selection."
                if args.validation_trials_per_class else
                f"Validation-selected LOSO: sessions {train_sessions} from 14 source subjects train; "
                f"source session {args.validation_session} validates; held-out subject sessions {args.sessions} test once after selection."
                if args.validation_session is not None else
                f"All-session LOSO: sessions {args.sessions} from 14 source subjects train; held-out subject sessions {args.sessions} evaluate.")
    result = {"schema_version": 2,
              "method": f"DMMR with {args.encoder.upper()} encoder, NPZ loader adaptation", "protocol": protocol,
              "subject": args.subject,
              "hyperparameters": {key: getattr(args, key) for key in ("time_steps", "pretrain_epochs", "finetune_epochs", "iteration", "batch_size", "lr", "weight_decay", "scheduler", "grad_clip", "beta", "seed", "checkpoint_every", "evaluate_test_every", "deterministic")},
              "provenance": {"artifact_directory": str(run_dir),
                             "metadata": str(run_dir / "run_metadata.json"),
                             "metrics": str(metrics_file),
                             "git_commit": provenance["git_commit"],
                             "source_sha256": provenance["source_sha256"]},
              "stage1_source": ({"checkpoint": str(stage1_source), "sha256": sha256(stage1_source)}
                                if stage1_source is not None else None),
              "encoder": args.encoder,
              "sessions": args.sessions, "train_sessions": train_sessions, "validation_session": args.validation_session,
              "validation_trials_per_class": args.validation_trials_per_class, "validation_split": split_manifest,
              "selection_metric": ("validation trial accuracy, then validation trial macro-F1, then validation sample accuracy"
                                   if val_loader is not None else "fixed final epoch; no validation and no test-time epoch selection"),
              "diagnostic_test_note": (f"Held-out test evaluated every {args.evaluate_test_every} epoch(s) for curve diagnosis only; never used for checkpoint selection."
                                       if args.evaluate_test_every else None),
              "pretrain_history": pretrain_history, "best_validation": best_validation,
              "validation_selected_test": selected_test, "fixed_final_test": final_test, "finetune_history": history}
    output = args.output_dir / f"subject_{args.subject:02d}.json"
    write_metrics_csv(run_dir / "epoch_metrics.csv", [*pretrain_history, *history])
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    atomic_torch_save({"pretrain": pretrain.state_dict(), "finetune": fine.state_dict(), "result": result}, output.with_suffix(".pt"))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
