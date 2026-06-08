from pathlib import Path

import pytest

from import_data import (
    ImportConfig,
    alternating_formats,
    build_parser,
    ensure_targets_are_writable,
)


def test_import_cli_defaults_to_paired_tables_and_1000_rows():
    args = build_parser().parse_args([])
    assert args.limit == 1000
    assert args.parquet_table == "flickr30k_parquet"
    assert args.vortex_table == "flickr30k_vortex"
    assert args.overwrite is False


def test_alternating_formats_switches_first_writer_each_batch():
    assert alternating_formats(0) == ("parquet", "vortex")
    assert alternating_formats(1) == ("vortex", "parquet")


def test_existing_target_requires_explicit_overwrite():
    with pytest.raises(RuntimeError, match="already exist"):
        ensure_targets_are_writable(
            existing={"flickr30k_parquet"}, overwrite=False
        )


def test_import_config_rejects_same_table_name(tmp_path: Path):
    with pytest.raises(ValueError, match="must be different"):
        ImportConfig(
            data_dir=tmp_path,
            jar_path=tmp_path / "lake.jar",
            parquet_table="same",
            vortex_table="same",
            limit=1000,
            batch_size=100,
            seed=7,
            overwrite=False,
            output=tmp_path / "import.json",
        )
