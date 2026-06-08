import pytest

from multimodal_data import (
    ImageRecord,
    Sample,
    canonical_record_digest,
    canonical_sample_digest,
    decode_rgb,
    expand_captions,
    select_filenames,
    split_filenames,
)


def test_select_filenames_is_stable_and_limit_zero_means_all():
    names = ["c.jpg", "a.jpg", "b.jpg", "d.jpg"]
    assert select_filenames(names, limit=0, seed=7) == tuple(sorted(names))
    assert select_filenames(names, limit=2, seed=7) == select_filenames(
        reversed(names), limit=2, seed=7
    )


def test_split_happens_by_filename_before_caption_expansion():
    names = [f"{index}.jpg" for index in range(20)]
    splits = split_filenames(names, seed=11)
    assert set(splits["train"]).isdisjoint(splits["val"])
    assert set(splits["train"]).isdisjoint(splits["test"])
    assert set(splits["val"]).isdisjoint(splits["test"])
    assert set().union(*map(set, splits.values())) == set(names)


def test_expand_captions_preserves_filename_and_blob():
    record = ImageRecord("1.jpg", b"jpeg", 10, 20, ("one", "two"), ())
    assert expand_captions(record) == (
        Sample("1.jpg", b"jpeg", "one"),
        Sample("1.jpg", b"jpeg", "two"),
    )


def test_canonical_digests_are_order_independent_and_content_sensitive():
    left = ImageRecord("1.jpg", b"a", 10, 20, ("one",), ())
    right = ImageRecord("2.jpg", b"b", 20, 10, ("two",), ())
    assert canonical_record_digest([left, right]) == canonical_record_digest(
        [right, left]
    )
    assert canonical_sample_digest(expand_captions(left)) != canonical_sample_digest(
        (Sample("1.jpg", b"a", "changed"),)
    )


def test_decode_rgb_raises_for_corrupt_bytes():
    with pytest.raises(ValueError, match="cannot decode image"):
        decode_rgb(b"not-an-image")
