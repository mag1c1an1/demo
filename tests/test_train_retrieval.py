from multimodal_data import Sample
from train_retrieval import LakeSoulSampleDataset, build_parser


def test_training_cli_selects_backend_and_table():
    args = build_parser().parse_args(
        ["--backend", "ray", "--table", "flickr30k_vortex", "--epochs", "1"]
    )
    assert args.backend == "ray"
    assert args.table == "flickr30k_vortex"
    assert args.epochs == 1


def test_dataset_decodes_samples(jpeg_bytes):
    dataset = LakeSoulSampleDataset(
        sample_factory=lambda: iter([Sample("a.jpg", jpeg_bytes, "caption")])
    )
    image, caption = next(iter(dataset))
    assert image.mode == "RGB"
    assert image.size == (8, 6)
    assert caption == "caption"
