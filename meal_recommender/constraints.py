"""Hard constraints: everything that must never be violated, enforced in code rather than in a prompt."""
from __future__ import annotations

import re
from typing import List

from .models import MenuItem, RequestConstraints, UserProfile

# A diet is satisfied if the item carries any of these labels.
DIET_ACCEPTS = {
    "vegan": {"vegan"},
    "vegetarian": {"vegetarian", "vegan"},
    "pescatarian": {"pescatarian", "vegetarian", "vegan"},
    "gluten_free": {"gluten_free"},
}

_PRICE_PATTERN = re.compile(
    r"(?:under|below|less than|max(?:imum)?|up to|no more than|<)\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:dollars|bucks|usd)?",
    re.IGNORECASE,
)
_EXCLUDE_PATTERN = re.compile(r"\b(?:no|without|hold the|not)\s+([a-z][a-z\- ]{1,30}?)(?=[,.;!?]|\band\b|$)", re.IGNORECASE)

# Words after "no"/"not" that describe style rather than an ingredient; these are handled by retrieval instead.
_STYLE_WORDS = {"too", "very", "so", "spicy", "greasy", "heavy", "fried", "sweet", "big", "much", "more"}


# Words that stand for a group of ingredients. "no cheese" must also exclude mozzarella, "no pork" must exclude
# bacon, and so on. A group can also map to tracked allergens, so "no nuts" excludes every dish labelled with them.
INGREDIENT_GROUPS = {
    "cheese": ({"cheese", "mozzarella", "parmesan", "cheddar", "feta", "paneer", "ricotta", "queso", "provolone",
                "halloumi", "cotija", "gruyere", "swiss"}, set()),
    "dairy": ({"cheese", "milk", "cream", "butter", "yogurt"}, {"dairy"}),
    "egg": ({"egg", "omelette", "mayo"}, {"eggs"}),
    "nut": ({"nut", "peanut", "cashew", "almond", "walnut", "pecan", "pistachio", "pesto"}, {"peanuts", "tree_nuts"}),
    "gluten": ({"wheat", "bread", "flour"}, {"wheat"}),
    "seafood": ({"fish", "salmon", "tuna", "shrimp", "prawn", "crab", "lobster", "clam", "mussel", "oyster",
                 "scallop", "squid", "calamari", "cod", "eel", "poke"}, {"fish", "shellfish"}),
    "pork": ({"pork", "bacon", "ham", "sausage", "pepperoni", "chorizo", "prosciutto", "spam", "carnitas",
              "pastor", "chashu", "pastrami"}, set()),
    "beef": ({"beef", "steak", "ribeye", "brisket", "bulgogi", "bolognese", "kofta", "pho bo"}, set()),
    "meat": ({"meat", "chicken", "beef", "pork", "lamb", "turkey", "bacon", "ham", "sausage", "pepperoni",
              "chorizo", "steak", "ribeye", "brisket", "spam", "carnitas", "pastor", "chashu", "pastrami", "bulgogi",
              "shawarma", "gyro", "kebab", "kofta", "bolognese", "wings", "souvlaki", "duck"}, set()),
}
_GROUP_ALIASES = {"cheeses": "cheese", "eggs": "egg", "nuts": "nut", "fish": "seafood", "shellfish": "seafood",
                  "red meat": "beef", "meats": "meat", "milk": "dairy"}
# Words that start like an ingredient but are not one ("butternut" squash is not butter, nutmeg is not a nut).
_FALSE_FRIENDS = re.compile(r"\b(?:butternut|nutmeg)\b")


def parse_request(text: str) -> RequestConstraints:
    """Extract hard limits stated in the request itself."""
    prices = [float(p) for p in _PRICE_PATTERN.findall(text or "")]
    excluded: List[str] = []
    for match in _EXCLUDE_PATTERN.findall(text or ""):
        term = match.strip().lower()
        if term and term.split()[0] not in _STYLE_WORDS:
            excluded.append(term)
    return RequestConstraints(max_price=min(prices) if prices else None, excluded_terms=excluded)


def _mentions(item: MenuItem, term: str) -> bool:
    text = _FALSE_FRIENDS.sub(" ", f"{item.dish_name} {item.description}".lower())
    term = term.strip().lower()
    group = INGREDIENT_GROUPS.get(_GROUP_ALIASES.get(term, term))
    if group:
        words, allergens = group
        if set(item.allergens) & allergens:
            return True
    else:
        words = {term}
    for word in words:
        singular = word[:-1] if word.endswith("s") and not word.endswith("ss") else word
        if re.search(rf"\b{re.escape(singular)}", text):
            return True
    return False


def violations(item: MenuItem, profile: UserProfile, request: RequestConstraints) -> List[str]:
    """Return every reason the item is not allowed (empty list = allowed)."""
    reasons: List[str] = []

    hit = sorted(set(item.allergens) & set(profile.allergies))
    if hit:
        reasons.append(f"contains allergen: {', '.join(hit)}")

    if profile.diet and not (set(item.dietary) & DIET_ACCEPTS[profile.diet]):
        reasons.append(f"not {profile.diet}")

    limit = min(profile.budget, request.max_price) if request.max_price is not None else profile.budget
    if item.price > limit:
        reasons.append(f"price ${item.price:.2f} over limit ${limit:.2f}")

    for term in list(profile.dislikes) + list(request.excluded_terms):
        if term and _mentions(item, term.lower()):
            reasons.append(f"mentions disliked/excluded '{term}'")

    return reasons


def filter_items(items: List[MenuItem], profile: UserProfile, request: RequestConstraints) -> List[MenuItem]:
    return [item for item in items if not violations(item, profile, request)]
