def test_runtime_capabilities_are_available():
    import daft
    import ray

    assert callable(daft.from_arrow)
    assert callable(daft.set_runner_native)
    assert callable(daft.set_runner_ray)
    decode_expr = daft.col("image_blob").decode_image(
        on_error="null",
        mode="RGB",
    )
    assert decode_expr is not None
    assert ray.__version__


def test_daft_native_runner_smoke():
    import subprocess
    import sys

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import daft; "
            "daft.set_runner_native(num_threads=1); "
            "assert daft.from_pydict({'value':[1,2]}).to_arrow().num_rows == 2",
        ],
        check=False,
    )
    assert completed.returncode == 0
