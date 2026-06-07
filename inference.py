"""Image-Text Retrieval on Flickr30k using CLIP."""
import io

import duckdb
import numpy as np
import torch
from lakesoul.arrow.dataset import lakesoul_dataset
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


def load_data(conn, limit: int | None = None):
    """Load images and captions from the LakeSoul table.

    Returns:
        image_bytes:  list of (filename, jpeg_bytes)
        captions:     list of (filename, caption_text)
    """
    limit_clause = f"LIMIT {limit}" if limit else ""
    rows = conn.sql(
        f"SELECT filename, image_blob, captions FROM flickr30k {limit_clause}"
    ).fetchall()

    image_bytes = []
    captions = []
    for filename, blob, caps in rows:
        image_bytes.append((filename, blob))
        for cap in caps:
            captions.append((filename, cap))
    return image_bytes, captions


def compute_image_embeddings(model, processor, image_bytes, batch_size=64):
    """Encode all images into normalized embeddings."""
    device = model.device
    all_embeddings = []
    all_filenames = []

    for i in tqdm(range(0, len(image_bytes), batch_size), desc="Encoding images"):
        batch = image_bytes[i : i + batch_size]
        filenames = [f for f, _ in batch]
        images = [Image.open(io.BytesIO(b)).convert("RGB") for _, b in batch]

        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            emb = model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)

        all_embeddings.append(emb.cpu().numpy())
        all_filenames.extend(filenames)

    return np.concatenate(all_embeddings, axis=0), all_filenames


def compute_text_embeddings(model, processor, captions, batch_size=128):
    """Encode all captions into normalized embeddings."""
    device = model.device
    all_embeddings = []
    all_filenames = []

    for i in tqdm(range(0, len(captions), batch_size), desc="Encoding texts"):
        batch = captions[i : i + batch_size]
        texts = [c for _, c in batch]
        fnames = [f for f, _ in batch]

        inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            emb = model.get_text_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)

        all_embeddings.append(emb.cpu().numpy())
        all_filenames.extend(fnames)

    return np.concatenate(all_embeddings, axis=0), all_filenames


def evaluate_recall(img_emb, img_names, txt_emb, txt_names, ks=(1, 5, 10)):
    """Evaluate Recall@K for both image→text and text→image retrieval.

    Each caption is treated as a query; the correct answer is its source image.
    """
    # Build filename → image index mapping
    img_name_to_idx = {name: i for i, name in enumerate(img_names)}

    # similarity: [num_texts, num_images]
    similarity = txt_emb @ img_emb.T

    # For each text query, find the rank of its ground-truth image
    gt_img_indices = np.array([img_name_to_idx[name] for name in txt_names])

    # Sort by similarity descending, get rank of each ground-truth
    ranks = np.zeros(len(txt_names), dtype=int)
    for i in range(len(txt_names)):
        sorted_indices = np.argsort(-similarity[i])  # descending
        rank = np.where(sorted_indices == gt_img_indices[i])[0][0] + 1
        ranks[i] = rank

    results = {}
    for k in ks:
        results[f"R@{k}"] = (ranks <= k).mean()

    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Connect and load data (lakesoul_dataset registers the table with duckdb)
    conn = duckdb.connect()
    lakesoul_dataset("flickr30k")
    image_bytes, captions = load_data(conn, limit=None)
    print(f"Loaded {len(image_bytes)} images, {len(captions)} captions (5 per image)")

    # Load CLIP model
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    print(f"Model: {model_name}")

    # Compute embeddings
    img_emb, img_names = compute_image_embeddings(model, processor, image_bytes)
    txt_emb, txt_names = compute_text_embeddings(model, processor, captions)

    # Evaluate
    results = evaluate_recall(img_emb, img_names, txt_emb, txt_names)
    print(f"\nText-to-Image Retrieval (treating each caption as a query):")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
