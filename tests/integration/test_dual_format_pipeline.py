import os

import pytest

from benchmark_report import write_reports
from data_backends import BackendConfig, iter_samples
from import_data import import_records
from multimodal_data import canonical_sample_digest

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.environ.get("RUN_LAKESOUL_INTEGRATION") != "1",
    reason="set RUN_LAKESOUL_INTEGRATION=1 to run LakeSoul integration",
)
def test_dual_format_backends_and_reports(spark_session, tiny_image_records, tmp_path):
    suffix = os.getpid()
    parquet_table = f"flickr30k_parquet_test_{suffix}"
    vortex_table = f"flickr30k_vortex_test_{suffix}"
    spark_session.sql(f"DROP TABLE IF EXISTS `{parquet_table}`")
    spark_session.sql(f"DROP TABLE IF EXISTS `{vortex_table}`")
    try:
        result = import_records(
            spark_session,
            tiny_image_records,
            parquet_table=parquet_table,
            vortex_table=vortex_table,
            batch_size=2,
        )
        assert result["parquet"]["digest"] == result["vortex"]["digest"]

        digests = {}
        for backend, runner in [
            ("native", "native"),
            ("ray", "native"),
            ("daft", "native"),
            ("daft", "ray"),
        ]:
            config = BackendConfig(
                backend,
                vortex_table,
                "all",
                2,
                7,
                0,
                "local",
                runner,
                False,
            )
            digests[f"{backend}-{runner}"] = canonical_sample_digest(
                iter_samples(config)
            )
        assert len(set(digests.values())) == 1
        assert write_reports([], {"integration": True}, tmp_path)
    finally:
        spark_session.sql(f"DROP TABLE IF EXISTS `{parquet_table}`")
        spark_session.sql(f"DROP TABLE IF EXISTS `{vortex_table}`")
