from src.geo.haversine import haversine_km
from src.geo.locations import LocationService
from src.utils.paths import PROJECT_ROOT


def test_delhi_mumbai_symmetric():
    service = LocationService(PROJECT_ROOT / "data" / "reference" / "airports.csv")
    delhi = service.resolve("Delhi")
    mumbai = service.resolve("Mumbai")
    a = haversine_km(delhi.latitude, delhi.longitude, mumbai.latitude, mumbai.longitude)
    b = haversine_km(mumbai.latitude, mumbai.longitude, delhi.latitude, delhi.longitude)
    assert abs(a - b) < 1e-6
    assert 1000 < a < 1500


def test_delhi_delhi_zero():
    service = LocationService(PROJECT_ROOT / "data" / "reference" / "airports.csv")
    delhi = service.resolve("Delhi")
    distance = haversine_km(delhi.latitude, delhi.longitude, delhi.latitude, delhi.longitude)
    assert distance == 0


def test_leh_delhi_positive():
    service = LocationService(PROJECT_ROOT / "data" / "reference" / "airports.csv")
    leh = service.resolve("Leh")
    delhi = service.resolve("Delhi")
    distance = service.distance_between(leh, delhi)
    assert distance > 0
    assert 400 < distance < 900


def test_distance_finite():
    value = haversine_km(28.5562, 77.1, 13.1986, 77.7066)
    assert value == value
    assert abs(value) != float("inf")
    assert value > 0


def test_unknown_city_does_not_invent_coordinates():
    from src.geo.locations import LocationNotFoundError

    service = LocationService(PROJECT_ROOT / "data" / "reference" / "airports.csv")
    try:
        service.resolve("NotARealCityXYZ")
        raise AssertionError("expected LocationNotFoundError")
    except LocationNotFoundError:
        pass
