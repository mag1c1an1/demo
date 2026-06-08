"""Fine-tune CLIP on Flickr30k through selectable LakeSoul data backends."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from data_backends import BackendConfig, iter_samples
from multimodal_data import Sample, decode_rgb


class LakeSoulSampleDataset(IterableDataset):
    def __init__(self, sample_factory: Callable[[], Iterator[Sample]]):
        super().__init__()
        self._sample_factory = sample_factory

    def __iter__(self):
        for sample in self._sample_factory():
            yield decode_rgb(sample.image_bytes), sample.caption


def make_dataset(args: argparse.Namespace, split: str) -> LakeSoulSampleDataset:
    config = BackendConfig(
        backend=args.backend,
        table_name=args.table,
        split=split,
        batch_size=args.data_batch_size,
        seed=args.seed,
        limit=args.limit,
        ray_address=args.ray_address,
        daft_runner=args.daft_runner,
        skip_corrupt=args.skip_corrupt,
    )
    return LakeSoulSampleDataset(lambda: iter_samples(config))


def collate_fn(batch, processor):
    images, texts = zip(*batch)
    image_inputs = processor(images=list(images), return_tensors="pt", padding=True)
    text_inputs = processor(
        text=list(texts), return_tensors="pt", padding=True, truncation=True
    )
    return image_inputs, text_inputs


def contrastive_loss(image_emb, text_emb, temperature=0.07):
    logits = (image_emb @ text_emb.T) * math.exp(temperature)
    labels = torch.arange(len(logits), device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2


def train_one_epoch(
    model, processor, dataloader, optimizer, scaler, device, temperature
):
    model.train()
    total_loss = 0.0
    batch_count = 0
    progress = tqdm(dataloader, desc="Training")
    for image_inputs, text_inputs in progress:
        image_inputs = {key: value.to(device) for key, value in image_inputs.items()}
        text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            image_embeddings = model.get_image_features(**image_inputs)
            text_embeddings = model.get_text_features(**text_inputs)
            image_embeddings = image_embeddings / image_embeddings.norm(
                dim=-1, keepdim=True
            )
            text_embeddings = text_embeddings / text_embeddings.norm(
                dim=-1, keepdim=True
            )
            loss = contrastive_loss(
                image_embeddings, text_embeddings, temperature
            )
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        batch_count += 1
        progress.set_postfix({"loss": f"{loss.item():.4f}"})
    if batch_count == 0:
        raise RuntimeError("training split produced no batches")
    return total_loss / batch_count


@torch.no_grad()
def evaluate_recall(model, processor, dataloader, device, ks=(1, 5, 10)):
    model.eval()
    all_image_embeddings = []
    all_text_embeddings = []
    for image_inputs, text_inputs in tqdm(dataloader, desc="Evaluating"):
        image_inputs = {key: value.to(device) for key, value in image_inputs.items()}
        text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
        image_embeddings = model.get_image_features(**image_inputs)
        text_embeddings = model.get_text_features(**text_inputs)
        image_embeddings = image_embeddings / image_embeddings.norm(
            dim=-1, keepdim=True
        )
        text_embeddings = text_embeddings / text_embeddings.norm(
            dim=-1, keepdim=True
        )
        all_image_embeddings.append(image_embeddings.cpu().numpy())
        all_text_embeddings.append(text_embeddings.cpu().numpy())
    if not all_image_embeddings:
        return {f"R@{k}": 0.0 for k in ks}
    image_embeddings = np.concatenate(all_image_embeddings, axis=0)
    text_embeddings = np.concatenate(all_text_embeddings, axis=0)
    similarity = text_embeddings @ image_embeddings.T
    ground_truth = np.arange(len(similarity))
    results = {}
    for k in ks:
        correct = sum(
            ground_truth[index] in np.argsort(-similarity[index])[:k]
            for index in range(len(similarity))
        )
        results[f"R@{k}"] = correct / len(similarity)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune CLIP retrieval from a LakeSoul backend"
    )
    parser.add_argument(
        "--backend", choices=["native", "ray", "daft"], default="native"
    )
    parser.add_argument("--table", default="flickr30k_vortex")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--ray-address", default="local")
    parser.add_argument(
        "--daft-runner", choices=["native", "ray"], default="native"
    )
    parser.add_argument("--skip-corrupt", action="store_true")
    parser.add_argument(
        "--model-name", default="openai/clip-vit-base-patch32"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("clip-flickr30k-finetuned")
    )
    parser.add_argument("--dry-run-batches", type=int, default=0)
    return parser


def _loader(
    args: argparse.Namespace,
    split: str,
    processor=None,
):
    collator = (
        (lambda batch: collate_fn(batch, processor))
        if processor is not None
        else (lambda batch: batch)
    )
    return DataLoader(
        make_dataset(args, split),
        batch_size=args.batch_size,
        collate_fn=collator,
        num_workers=0,
    )


def _dry_run(args: argparse.Namespace) -> int:
    loader = _loader(args, "train")
    count = 0
    for batch in loader:
        images, captions = zip(*batch)
        if not all(image.mode == "RGB" for image in images):
            raise RuntimeError("dry-run batch contains non-RGB images")
        count += 1
        print(
            f"dry-run batch {count}: {len(images)} images, "
            f"{len(captions)} captions"
        )
        if count >= args.dry_run_batches:
            break
    if count < args.dry_run_batches:
        raise RuntimeError(
            f"requested {args.dry_run_batches} dry-run batches, got {count}"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run_batches < 0:
        raise ValueError("dry-run batches must be non-negative")
    if args.dry_run_batches:
        return _dry_run(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temperature = math.log(1 / 0.07)
    model = CLIPModel.from_pretrained(
        args.model_name, use_safetensors=True
    ).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)
    train_loader = _loader(args, "train", processor)
    val_loader = _loader(args, "val", processor)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01
    )
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(
            model,
            processor,
            train_loader,
            optimizer,
            scaler,
            device,
            temperature,
        )
        print(f"Train loss: {train_loss:.4f}")
        validation = evaluate_recall(
            model, processor, val_loader, device
        )
        print("Validation:", " ".join(f"{k}={v:.4f}" for k, v in validation.items()))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    test_loader = _loader(args, "test", processor)
    test_results = evaluate_recall(model, processor, test_loader, device)
    print("Test:", " ".join(f"{k}={v:.4f}" for k, v in test_results.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
