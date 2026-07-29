import fasterrag


def test_version_is_exported() -> None:
    assert fasterrag.__version__
    assert "__version__" in fasterrag.__all__
