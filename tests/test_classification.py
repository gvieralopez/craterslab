from craterslab.classification import SurfaceType


def test_surface_type_names() -> None:
    assert str(SurfaceType.COMPLEX_CRATER) == "Complex crater"
    assert str(SurfaceType.UNKNOWN) == "Unknown"
