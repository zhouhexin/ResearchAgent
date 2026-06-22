"""Train the lightweight MLP used by SAGE-style semantic segmentation."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from densex.corpus import read_jsonl
from retrieval.embed import Embedder
from sage_segmenter.model import SegmentationMLP, build_pair_features


DEFAULT_ANGLE_MODEL = "WhereIsAI/UAE-Large-V1"


def _device_arg(value: str) -> str:
    if value != "auto":
        return value
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _split_rows(rows: list[dict], validation_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)
    if validation_ratio <= 0 or len(shuffled) < 2:
        return shuffled, []
    validation_count = max(1, int(len(shuffled) * validation_ratio))
    return shuffled[validation_count:], shuffled[:validation_count]


def _features_and_labels(rows: list[dict], embedder: Embedder, device: str):
    import torch

    left = embedder.encode([row["s1"] for row in rows])
    right = embedder.encode([row["s2"] for row in rows])
    x1 = torch.from_numpy(left).to(device)
    x2 = torch.from_numpy(right).to(device)
    features = build_pair_features(x1, x2)
    labels = torch.tensor([float(row["label"]) for row in rows], dtype=torch.float32, device=device)
    return features, labels


def _evaluate(model: SegmentationMLP, features, labels, *, loss_name: str) -> dict:
    import torch

    model.eval()
    with torch.no_grad():
        logits = model(features)
        probabilities = torch.sigmoid(logits)
        mse = torch.mean((probabilities - labels) ** 2).item()
        predictions = (probabilities >= 0.5).float()
        accuracy = torch.mean((predictions == labels).float()).item()
        if loss_name == "bce":
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels).item()
        else:
            loss = mse
    return {"loss": loss, "mse": mse, "accuracy": accuracy, "count": int(labels.numel())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAGE semantic segmentation MLP")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--validation-pairs", type=Path, default=None)
    parser.add_argument("--embedding-model", default=DEFAULT_ANGLE_MODEL)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models" / "sage_segmenter_angle")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--loss", choices=["mse", "bce"], default="mse")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    import torch

    device = _device_arg(args.device)
    torch.manual_seed(args.seed)
    rows = read_jsonl(args.pairs)
    if not rows:
        raise RuntimeError(f"No training pairs found in {args.pairs}")
    if args.validation_pairs:
        train_rows = rows
        validation_rows = read_jsonl(args.validation_pairs)
    else:
        train_rows, validation_rows = _split_rows(rows, args.validation_ratio, args.seed)
    if not train_rows:
        raise RuntimeError("No training rows available after validation split")

    embedder = Embedder(args.embedding_model)
    train_features, train_labels = _features_and_labels(train_rows, embedder, device)
    validation_features = validation_labels = None
    if validation_rows:
        validation_features, validation_labels = _features_and_labels(validation_rows, embedder, device)

    model = SegmentationMLP(
        input_dim=train_features.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    indices = list(range(train_features.shape[0]))

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.Random(args.seed + epoch).shuffle(indices)
        epoch_loss = 0.0
        for start in range(0, len(indices), args.batch_size):
            batch_indices = torch.tensor(indices[start : start + args.batch_size], dtype=torch.long, device=device)
            batch_features = train_features.index_select(0, batch_indices)
            batch_labels = train_labels.index_select(0, batch_indices)
            logits = model(batch_features)
            if args.loss == "bce":
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch_labels)
            else:
                probabilities = torch.sigmoid(logits)
                loss = torch.nn.functional.mse_loss(probabilities, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(batch_indices)
        print(f"epoch={epoch} train_loss={epoch_loss / len(indices):.6f}")

    metrics = {"train": _evaluate(model, train_features, train_labels, loss_name=args.loss)}
    if validation_features is not None and validation_labels is not None:
        metrics["validation"] = _evaluate(model, validation_features, validation_labels, loss_name=args.loss)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_dir / "mlp.pt")
    config = {
        "embedding_model": args.embedding_model,
        "input_dim": int(train_features.shape[1]),
        "embedding_dim": int(train_features.shape[1] // 4),
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "loss": args.loss,
        "threshold_default": 0.55,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote model to {args.output_dir}")


if __name__ == "__main__":
    main()
