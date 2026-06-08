from data_backends import (
    BackendConfig,
    build_daft_worker_command,
    iter_samples_from_records,
)
from multimodal_data import ImageRecord, canonical_sample_digest


def records():
    return [
        ImageRecord("a.jpg", b"a", 1, 1, ("a1", "a2"), ()),
        ImageRecord("b.jpg", b"b", 1, 1, ("b1",), ()),
        ImageRecord("c.jpg", b"c", 1, 1, ("c1",), ()),
    ]


def test_native_backend_splits_before_caption_expansion():
    config = BackendConfig(
        backend="native",
        table_name="unused",
        split="train",
        batch_size=2,
        seed=3,
        limit=0,
        ray_address="local",
        daft_runner="native",
        skip_corrupt=False,
    )
    samples = tuple(iter_samples_from_records(records(), config))
    selected_names = {sample.filename for sample in samples}
    assert all(
        (record.filename in selected_names)
        == bool(set(record.captions) & {sample.caption for sample in samples})
        for record in records()
    )


def test_backend_digest_does_not_depend_on_input_order():
    config = BackendConfig(
        "native", "unused", "all", 2, 3, 0, "local", "native", False
    )
    forward = iter_samples_from_records(records(), config)
    reverse = iter_samples_from_records(reversed(records()), config)
    assert canonical_sample_digest(forward) == canonical_sample_digest(reverse)


def test_build_daft_worker_command_contains_runner_and_table():
    command = build_daft_worker_command(
        BackendConfig(
            "daft",
            "flickr30k_vortex",
            "train",
            32,
            7,
            1000,
            "ray://head:10001",
            "ray",
            False,
        )
    )
    assert command[-4:] == [
        "--runner",
        "ray",
        "--table",
        "flickr30k_vortex",
    ]
    assert "--ray-address" in command


def test_daft_arrow_transform_decodes_and_expands(jpeg_bytes):
    import pyarrow as pa

    from daft_worker import transform_arrow

    table = pa.table(
        {
            "filename": ["a.jpg"],
            "image_blob": [jpeg_bytes],
            "width": [8],
            "height": [6],
            "captions": [["one", "two"]],
            "bboxes": [[]],
        }
    )
    batches = list(transform_arrow(table, skip_corrupt=False))
    rows = pa.Table.from_batches(batches).to_pylist()
    assert [row["caption"] for row in rows] == ["one", "two"]
    assert all(row["decoded_width"] == 8 for row in rows)
    assert all(row["decoded_height"] == 6 for row in rows)
