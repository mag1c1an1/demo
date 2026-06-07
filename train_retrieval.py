"""Fine-tune CLIP on Flickr30k for Image-Text Retrieval (Contrastive Learning)."""

import io
import math
import random
from collections import defaultdict

import duckdb
import numpy as np
import torch
import torch.nn.functional as F
from lakesoul.arrow.dataset import lakesoul_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# ────────────────────────────── Dataset ──────────────────────────────


class Flickr30kDataset(Dataset):
    """Pairs of (image, caption) loaded lazily from LakeSoul.

    Each image appears 5 times (once per caption). For contrastive learning,
    the diagonal of the batch is the positive pair.

    Images are loaded on-the-fly in __getitem__ to avoid loading 31k blobs
    into memory at once.
    """

    def __init__(
        self,
        processor,
        split_ratio: tuple[float, float, float] = (0.937, 0.0315, 0.0315),
    ):
        # Only fetch filenames — no blobs
        # lakesoul_dataset registers the table on a global/default duckdb connection
        self._dataset = lakesoul_dataset("flickr30k")
        duckdb.register("flickr30k", self._dataset)
        rows = duckdb.sql("SELECT filename, captions FROM flickr30k").fetchall()
        random.shuffle(rows)

        # Create (filename, caption) pairs — lightweight, ~158k tuples
        self.pairs: list[tuple[str, str]] = []
        for filename, caps in rows:
            for cap in caps:
                self.pairs.append((filename, cap))

        # Karpathy-style split
        n = len(self.pairs)
        train_end = int(n * split_ratio[0])
        val_end = train_end + int(n * split_ratio[1])

        self.splits = {
            "train": self.pairs[:train_end],
            "val": self.pairs[train_end:val_end],
            "test": self.pairs[val_end:],
        }
        self.processor = processor

    def set_split(self, split: str):
        self._data = self.splits[split]
        return self

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):  # type: ignore[override]
        filename, caption = self._data[idx]
        print(filename)
        sql = f"select image_blob from flickr30k where filename = '{filename}'"
        print(sql)
        # Lazy load: fetch single image blob by filename
        blob = duckdb.sql(sql).fetchone()[0]
        image = Image.open(io.BytesIO(blob)).convert("RGB")
        return image, caption


def collate_fn(batch, processor):
    """Collate images and texts into a batch, returning processed tensors."""
    images, texts = zip(*batch)
    image_inputs = processor(images=list(images), return_tensors="pt", padding=True)
    text_inputs = processor(
        text=list(texts), return_tensors="pt", padding=True, truncation=True
    )
    return image_inputs, text_inputs


# ────────────────────────────── Loss ──────────────────────────────


def contrastive_loss(image_emb, text_emb, temperature=0.07):
    """CLIP-style InfoNCE loss.

    image_emb: [B, D]  normalized
    text_emb:  [B, D]  normalized
    Returns: scalar loss (image→text + text→image cross-entropy) / 2
    """
    # Cosine similarity logits scaled by temperature
    logits = (image_emb @ text_emb.T) * math.exp(temperature)  # [B, B]

    # Ground truth: diagonal (image_i ↔ text_i)
    labels = torch.arange(len(logits), device=logits.device)

    loss_i2t = F.cross_entropy(logits, labels)  # image → text
    loss_t2i = F.cross_entropy(logits.T, labels)  # text → image

    return (loss_i2t + loss_t2i) / 2


# ────────────────────────────── Training ──────────────────────────────


def train_one_epoch(
    model, processor, dataloader, optimizer, scaler, device, temperature
):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc="Training")
    for image_inputs, text_inputs in pbar:
        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            img_emb = model.get_image_features(**image_inputs)
            txt_emb = model.get_text_features(**text_inputs)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
            loss = contrastive_loss(img_emb, txt_emb, temperature)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate_recall(model, processor, dataloader, device, ks=(1, 5, 10)):
    """Recall@K on validation set. All pairs are used as queries."""
    model.eval()

    all_img_emb = []
    all_txt_emb = []
    for image_inputs, text_inputs in tqdm(dataloader, desc="Evaluating"):
        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        img_emb = model.get_image_features(**image_inputs)
        txt_emb = model.get_text_features(**text_inputs)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

        all_img_emb.append(img_emb.cpu().numpy())
        all_txt_emb.append(txt_emb.cpu().numpy())

    img_emb = np.concatenate(all_img_emb, axis=0)
    txt_emb = np.concatenate(all_txt_emb, axis=0)

    # similarity: [N, N] — diagonal is the positive pair
    similarity = txt_emb @ img_emb.T
    gt = np.arange(len(similarity))

    results = {}
    for k in ks:
        correct = 0
        for i in range(len(similarity)):
            top_k = np.argsort(-similarity[i])[:k]
            if gt[i] in top_k:
                correct += 1
        results[f"R@{k}"] = correct / len(similarity)

    return results


# ────────────────────────────── Main ──────────────────────────────


def main():
    # Config
    model_name = "openai/clip-vit-base-patch32"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    epochs = 5
    lr = 1e-5
    temperature = math.log(1 / 0.07)  # init logit_scale ≈ 0.07
    use_amp = device.type == "cuda"

    print(f"Device: {device} | Batch: {batch_size} | LR: {lr} | Epochs: {epochs}")

    # Load model
    model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    # Load data from LakeSoul (lakesoul_dataset auto-registers on default duckdb connection)
    dataset = Flickr30kDataset(processor)

    # Dataloaders
    train_loader = DataLoader(
        dataset.set_split("train"),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, processor),
        num_workers=0,  # DuckDB connections can't be shared across processes
    )
    val_loader = DataLoader(
        dataset.set_split("val"),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, processor),
        num_workers=0,
    )

    print(
        f"Train: {len(dataset.splits['train'])} pairs | Val: {len(dataset.splits['val'])} | Test: {len(dataset.splits['test'])}"
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Train
    for epoch in range(1, epochs + 1):
        print(f"\n─── Epoch {epoch}/{epochs} ───")
        train_loss = train_one_epoch(
            model, processor, train_loader, optimizer, scaler, device, temperature
        )
        print(f"Train loss: {train_loss:.4f}")

        val_results = evaluate_recall(model, processor, val_loader, device)
        print("Val:", "  ".join(f"{k}: {v:.4f}" for k, v in val_results.items()))

    # Save
    model.save_pretrained("clip-flickr30k-finetuned")
    processor.save_pretrained("clip-flickr30k-finetuned")
    print("\nSaved to clip-flickr30k-finetuned/")

    # Final test
    test_loader = DataLoader(
        dataset.set_split("test"),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, processor),
        num_workers=0,
    )
    test_results = evaluate_recall(model, processor, test_loader, device)
    print("\n─── Test Results ───")
    for k, v in test_results.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
