from __future__ import annotations

import hashlib
import io
import json
import random
from dataclasses import dataclass
from typing import Iterable, Mapping, TypedDict

from PIL import Image

DEFAULT_SPLIT_RATIO = (0.937, 0.0315, 0.0315)


@dataclass(frozen=True)
class BoundingBox:
    chain_ids: tuple[str, ...]
    xmin: int
    ymin: int
    xmax: int
    ymax: int


@dataclass(frozen=True)
class ImageRecord:
    filename: str
    image_bytes: bytes
    width: int
    height: int
    captions: tuple[str, ...]
    bboxes: tuple[BoundingBox, ...]


@dataclass(frozen=True)
class Sample:
    filename: str
    image_bytes: bytes
    caption: str


class BoundingBoxPayload(TypedDict):
    chain_ids: list[str]
    xmin: int
    ymin: int
    xmax: int
    ymax: int


class RecordPayload(TypedDict):
    filename: str
    image_sha256: str
    width: int
    height: int
    captions: list[str]
    bboxes: list[BoundingBoxPayload]


def select_filenames(
    filenames: Iterable[str], *, limit: int, seed: int
) -> tuple[str, ...]:
    if limit < 0:
        raise ValueError("limit must be non-negative")
    ordered = sorted(set(filenames))
    if limit == 0 or limit >= len(ordered):
        return tuple(ordered)
    selected = random.Random(seed).sample(ordered, limit)
    return tuple(sorted(selected))


def split_filenames(
    filenames: Iterable[str],
    *,
    seed: int,
    ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIO,
) -> Mapping[str, tuple[str, ...]]:
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("split ratios must contain three values summing to 1")
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("split ratios must be non-negative")
    ordered = sorted(set(filenames))
    random.Random(seed).shuffle(ordered)
    train_end = int(len(ordered) * ratios[0])
    val_end = train_end + int(len(ordered) * ratios[1])
    return {
        "train": tuple(ordered[:train_end]),
        "val": tuple(ordered[train_end:val_end]),
        "test": tuple(ordered[val_end:]),
    }


def expand_captions(record: ImageRecord) -> tuple[Sample, ...]:
    return tuple(
        Sample(record.filename, record.image_bytes, caption)
        for caption in record.captions
    )


def decode_rgb(image_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
        return image.convert("RGB")
    except Exception as exc:
        raise ValueError("cannot decode image") from exc


def _record_payload(record: ImageRecord) -> RecordPayload:
    boxes: list[BoundingBoxPayload] = [
        BoundingBoxPayload(
            chain_ids=list(box.chain_ids),
            xmin=box.xmin,
            ymin=box.ymin,
            xmax=box.xmax,
            ymax=box.ymax,
        )
        for box in record.bboxes
    ]
    return {
        "filename": record.filename,
        "image_sha256": hashlib.sha256(record.image_bytes).hexdigest(),
        "width": record.width,
        "height": record.height,
        "captions": list(record.captions),
        "bboxes": boxes,
    }


def canonical_record_payload(record: ImageRecord) -> bytes:
    return json.dumps(
        _record_payload(record), sort_keys=True, separators=(",", ":")
    ).encode()


def canonical_record_digest_from_payloads(
    payloads: Iterable[tuple[str, bytes]],
) -> str:
    digest = hashlib.sha256()
    for _, payload in sorted(payloads):
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def canonical_record_digest(records: Iterable[ImageRecord]) -> str:
    return canonical_record_digest_from_payloads(
        (record.filename, canonical_record_payload(record)) for record in records
    )


def canonical_sample_digest_from_keys(
    keys: Iterable[tuple[str, str, str]],
) -> str:
    digest = hashlib.sha256()
    for key in sorted(keys):
        digest.update(json.dumps(key, separators=(",", ":")).encode())
    return digest.hexdigest()


def canonical_sample_digest(samples: Iterable[Sample]) -> str:
    return canonical_sample_digest_from_keys(
        (
            sample.filename,
            hashlib.sha256(sample.image_bytes).hexdigest(),
            sample.caption,
        )
        for sample in samples
    )
