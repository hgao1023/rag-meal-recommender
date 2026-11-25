"""Links that let the user act on a recommendation.

The menu is sample data with fictional restaurants, so instead of linking to a restaurant page we
search for the recommended dish near the user's ZIP code on real services.
"""
from __future__ import annotations

import re
from typing import Dict
from urllib.parse import urlencode

from .location import DEFAULT_ZIP
from .models import MenuItem


def search_terms(item: MenuItem) -> str:
    return f"{item.dish_name} {item.cuisine}" if item.cuisine.lower() not in item.dish_name.lower() else item.dish_name


def order_links(item: MenuItem, zip_code: str = DEFAULT_ZIP) -> Dict[str, str]:
    zip_code = zip_code if re.fullmatch(r"\d{5}", zip_code or "") else DEFAULT_ZIP
    terms = search_terms(item)
    return {
        "Google Maps": "https://www.google.com/maps/search/?" + urlencode({"api": 1, "query": f"{terms} near {zip_code}"}),
        "Yelp": "https://www.yelp.com/search?" + urlencode({"find_desc": terms, "find_loc": zip_code}),
    }
