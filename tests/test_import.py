import citrus


def test_version():
    assert isinstance(citrus.__version__, str)
    assert citrus.__version__
