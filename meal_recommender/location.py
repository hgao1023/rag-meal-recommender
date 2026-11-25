"""Turn what the user types as a location into a real US ZIP code.

Accepted input:
- a ZIP code: "95053" or "95053-1234" (digits only, nothing else),
- a full address ending in a ZIP code: "500 El Camino Real, Santa Clara, CA 95053",
- a place name: "NYU", "Golden Gate Park". Looked up with OpenStreetMap's free Nominatim service.

Every ZIP code is checked against an offline list of real US ZIP codes (the `zipcodes` package),
so "99999" is rejected and "10012" is shown as "New York, NY".
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from typing import Callable, Dict, List, Optional

import zipcodes
from pydantic import BaseModel

# A few West Coast cities with a central ZIP code, offered as a quick pick in the app.
CITY_ZIPS: Dict[str, str] = {
    "San Jose, CA": "95113",
    "Santa Clara, CA": "95050",
    "Sunnyvale, CA": "94086",
    "Mountain View, CA": "94041",
    "Palo Alto, CA": "94301",
    "Cupertino, CA": "95014",
    "Fremont, CA": "94538",
    "San Francisco, CA": "94103",
    "Oakland, CA": "94612",
    "Berkeley, CA": "94704",
    "Sacramento, CA": "95814",
    "Los Angeles, CA": "90012",
    "San Diego, CA": "92101",
    "Portland, OR": "97204",
    "Seattle, WA": "98101",
}
DEFAULT_CITY = "San Jose, CA"
DEFAULT_ZIP = CITY_ZIPS[DEFAULT_CITY]

_ZIP_ONLY = re.compile(r"^(\d{5})(?:-\d{4})?$")
# A ZIP code inside an address: a standalone 5-digit token, optionally ZIP+4, not glued to letters or digits.
_ZIP_IN_ADDRESS = re.compile(r"(?<![\w-])(\d{5})(?:-\d{4})?(?![\w-])")
_HAS_LETTER = re.compile(r"[^\W\d_]")

NOMINATIM_URL = "https://nominatim.openstreetmap.org"
USER_AGENT = "rag-meal-recommender/1.0 (demo app)"

Fetch = Callable[[str], List[dict] | dict]


class LocationResult(BaseModel):
    """Either a valid `zip_code` (with a readable `label`) or an `error` code explaining what is wrong."""
    zip_code: Optional[str] = None
    label: Optional[str] = None
    error: Optional[str] = None  # "empty" | "format" | "unknown_zip" | "not_found" | "lookup_failed"
    source: Optional[str] = None  # "zip" | "address" | "place"

    @property
    def ok(self) -> bool:
        return self.zip_code is not None


def zip_info(zip_code: str) -> Optional[str]:
    """"City, ST" for a real US ZIP code, None if the ZIP code does not exist."""
    try:
        matches = zipcodes.matching(zip_code)
    except (TypeError, ValueError):
        return None
    if not matches:
        return None
    m = matches[0]
    return f"{m['city'].title()}, {m['state']}"


def city_for_zip(zip_code: str) -> Optional[str]:
    return next((city for city, z in CITY_ZIPS.items() if z == zip_code), None)


def _http_get_json(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Language": "en"})
    with urllib.request.urlopen(request, timeout=8) as response:
        return json.loads(response.read().decode("utf-8"))


def _postcode(address: dict) -> Optional[str]:
    match = re.match(r"^\s*(\d{5})", str(address.get("postcode", "")))
    return match.group(1) if match else None


def geocode_place(name: str, fetch: Fetch = _http_get_json) -> Optional[tuple[str, str]]:
    """Look up a US place name. Returns (zip_code, place label) or None if nothing usable was found.

    Raises on network errors so the caller can tell "not found" from "lookup failed".
    """
    query = urllib.parse.urlencode({"q": name, "countrycodes": "us", "format": "jsonv2",
                                    "addressdetails": 1, "limit": 5})
    results = fetch(f"{NOMINATIM_URL}/search?{query}") or []
    if not results:
        return None
    # Trust the best match only: scanning further down for any result with a postcode can jump to a
    # same-named place in another state.
    first = results[0]
    zip_code = _postcode(first.get("address", {}))
    if zip_code:
        return zip_code, first.get("name") or name
    # Large areas (a city, a park) often have no single postcode: use the ZIP code at the area's centre.
    reverse = urllib.parse.urlencode({"lat": first["lat"], "lon": first["lon"], "format": "jsonv2",
                                      "addressdetails": 1, "zoom": 18})
    zip_code = _postcode(fetch(f"{NOMINATIM_URL}/reverse?{reverse}").get("address", {}))
    return (zip_code, first.get("name") or name) if zip_code else None


def _from_zip(zip_code: str, source: str, place: Optional[str] = None) -> LocationResult:
    city = zip_info(zip_code)
    if city is None:
        return LocationResult(error="unknown_zip", source=source)
    label = f"{place} · {city}" if place and place.lower() != city.split(",")[0].lower() else city
    return LocationResult(zip_code=zip_code, label=label, source=source)


def resolve_location(text: str, fetch: Optional[Fetch] = _http_get_json) -> LocationResult:
    """Validate a ZIP code, read the ZIP code from an address, or look up a place name.

    `fetch=None` disables the online place lookup.
    """
    text = (text or "").strip()
    if not text:
        return LocationResult(error="empty")

    if not _HAS_LETTER.search(text):
        # Only digits and symbols: it must be exactly a ZIP code. "95053!!" or "1234567" are errors.
        match = _ZIP_ONLY.match(text)
        return _from_zip(match.group(1), "zip") if match else LocationResult(error="format", source="zip")

    zips = _ZIP_IN_ADDRESS.findall(text)
    if zips:
        return _from_zip(zips[-1], "address")  # the ZIP code comes last in a US address

    if fetch is None:
        return LocationResult(error="not_found", source="place")
    try:
        found = geocode_place(text, fetch)
    except Exception:  # network error, rate limit, unexpected response
        return LocationResult(error="lookup_failed", source="place")
    if not found:
        return LocationResult(error="not_found", source="place")
    return _from_zip(found[0], "place", place=found[1])
