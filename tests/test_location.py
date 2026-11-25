import pytest

from meal_recommender.location import CITY_ZIPS, geocode_place, resolve_location, zip_info


def no_network(url):
    raise AssertionError("should not call the network")


@pytest.mark.parametrize("text, zip_code", [
    ("95053", "95053"),
    (" 95053 ", "95053"),
    ("95053-1234", "95053"),
    ("10012", "10012"),
    ("500 El Camino Real, Santa Clara, CA 95053", "95053"),
    ("10101 N De Anza Blvd, Cupertino, CA 95014", "95014"),   # 5-digit street number, the ZIP comes last
])
def test_valid_zip_and_address(text, zip_code):
    result = resolve_location(text, fetch=no_network)
    assert result.ok and result.zip_code == zip_code


@pytest.mark.parametrize("text, error", [
    ("", "empty"),
    ("   ", "empty"),
    ("1234567", "format"),     # too long
    ("9505", "format"),        # too short
    ("95053!!", "format"),     # symbols around a ZIP
    ("!95053?", "format"),
    ("@#$%", "format"),
    ("95-053", "format"),
    ("99999", "unknown_zip"),  # right format, not a real ZIP code
    ("00000", "unknown_zip"),
])
def test_invalid_zip_explains_why(text, error):
    result = resolve_location(text, fetch=no_network)
    assert not result.ok and result.error == error


def test_zip_info_gives_city():
    assert zip_info("10012") == "New York, NY"
    assert zip_info("99999") is None
    assert all(zip_info(z) for z in CITY_ZIPS.values())


def fake_osm(search, reverse=None):
    def fetch(url):
        return search if "/search?" in url else reverse
    return fetch


def test_place_name_with_postcode():
    fetch = fake_osm([{"name": "New York University", "lat": "40.7", "lon": "-73.9", "address": {"postcode": "10003"}}])
    result = resolve_location("NYU", fetch=fetch)
    assert result.ok and result.zip_code == "10003" and result.source == "place"
    assert result.label == "New York University · New York, NY"


def test_place_without_postcode_uses_reverse_lookup():
    fetch = fake_osm([{"name": "Golden Gate Park", "lat": "37.77", "lon": "-122.48", "address": {}}],
                     {"address": {"postcode": "94122"}})
    assert geocode_place("Golden Gate Park", fetch) == ("94122", "Golden Gate Park")


def test_place_not_found_and_lookup_failure():
    assert resolve_location("qwzxv plorp", fetch=fake_osm([])).error == "not_found"

    def broken(url):
        raise TimeoutError
    assert resolve_location("NYU", fetch=broken).error == "lookup_failed"
    assert resolve_location("NYU", fetch=None).error == "not_found"
