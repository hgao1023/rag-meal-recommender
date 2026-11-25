"""Rule-based ranking and template explanations (used when no LLM is configured, and as the LLM fallback)."""
from __future__ import annotations

import re
from typing import Dict, List

from .i18n import label
from .models import MenuItem, Recommendation, RequestConstraints, UserProfile
from .retriever import ScoredItem

# Request words -> menu tag they point to, used to explain *why* an item matches.
INTENT_TAGS: Dict[str, str] = {
    "light": "light", "lighter": "light", "refreshing": "light", "fresh": "light",
    "healthy": "healthy", "clean": "healthy", "nutritious": "healthy",
    "protein": "high_protein", "filling": "high_protein", "gym": "high_protein",
    "spicy": "spicy", "kick": "spicy",
    "soup": "soup", "soupy": "soup", "broth": "soup", "warm": "soup",
    "noodle": "noodles", "noodles": "noodles", "ramen": "noodles", "pho": "noodles",
    "salad": "salad", "greens": "salad",
    "quick": "quick", "fast": "quick", "snack": "quick",
    "comfort": "comfort", "cozy": "comfort", "hearty": "comfort",
    "breakfast": "breakfast", "brunch": "breakfast", "morning": "breakfast",
    "sweet": "sweet", "dessert": "dessert",
    "sandwich": "sandwich", "wrap": "sandwich",
    "rice": "rice",
    "grilled": "grilled", "bbq": "grilled",
    "keto": "low_carb", "carb": "low_carb",
}
TAG_PHRASES = {
    "light": "light", "healthy": "healthy", "high_protein": "high in protein", "spicy": "spicy",
    "soup": "a soup", "noodles": "a noodle dish", "salad": "a salad", "quick": "quick to eat",
    "comfort": "comfort food", "breakfast": "a breakfast dish", "sweet": "sweet", "dessert": "a dessert",
    "sandwich": "a sandwich or wrap", "rice": "a rice dish", "grilled": "grilled", "low_carb": "low in carbs",
}
GREASY_WORDS = ("greasy", "oily", "heavy")


def requested_tags(request_text: str) -> List[str]:
    words = re.findall(r"[a-z]+", request_text.lower())
    tags: List[str] = []
    for w in words:
        tag = INTENT_TAGS.get(w)
        if tag and tag not in tags:
            tags.append(tag)
    return tags


def explain(item: MenuItem, profile: UserProfile, request_text: str, constraints: RequestConstraints,
            lang: str = "en") -> str:
    matched = [t for t in requested_tags(request_text) if t in item.tags]
    not_fried = any(w in request_text.lower() for w in GREASY_WORDS) and "fried" not in item.tags
    limit = min(profile.budget, constraints.max_price) if constraints.max_price is not None else profile.budget

    if lang == "zh":
        parts = []
        if matched:
            parts.append("符合你要的：" + "、".join(label(t, "zh") for t in matched))
        if not_fried:
            parts.append("不是油炸的")
        parts.append(f"${item.price:.2f}，在你 ${limit:.0f} 的预算内")
        if profile.allergies:
            parts.append("不含你列出的过敏原（" + "、".join(label(a, "zh") for a in profile.allergies) + "）")
        if profile.diet:
            parts.append(f"属于{label(profile.diet, 'zh')}")
        return "；".join(parts) + "。"

    parts: List[str] = []
    if matched:
        parts.append("It is " + ", ".join(TAG_PHRASES[t] for t in matched) + ", as you asked")
    if not_fried:
        parts.append("it is not fried")
    parts.append(f"at ${item.price:.2f} it fits your ${limit:.0f} limit")
    if profile.allergies:
        parts.append("it contains none of your listed allergens (" + ", ".join(profile.allergies) + ")")
    if profile.diet:
        parts.append(f"it is {profile.diet.replace('_', '-')}")
    sentence = "; ".join(parts)
    return sentence[0].upper() + sentence[1:] + "."


def diversify(scored: List[ScoredItem], max_per_restaurant: int = 2) -> List[ScoredItem]:
    """Keep ranking order but avoid filling the list with one restaurant."""
    counts: Dict[str, int] = {}
    kept, overflow = [], []
    for item, score in scored:
        if counts.get(item.restaurant, 0) < max_per_restaurant:
            counts[item.restaurant] = counts.get(item.restaurant, 0) + 1
            kept.append((item, score))
        else:
            overflow.append((item, score))
    return kept + overflow


def rank_with_rules(scored: List[ScoredItem], profile: UserProfile, request_text: str,
                    constraints: RequestConstraints, top_n: int = 5, lang: str = "en") -> List[Recommendation]:
    """Order by semantic similarity plus a small bonus per requested tag the item actually has."""
    wanted = requested_tags(request_text)
    avoid_fried = any(w in request_text.lower() for w in GREASY_WORDS)

    def score(pair: ScoredItem) -> float:
        item, sim = pair
        bonus = 0.05 * sum(1 for t in wanted if t in item.tags)
        penalty = 0.1 if avoid_fried and "fried" in item.tags else 0.0
        return sim + bonus - penalty

    ordered = diversify(sorted(scored, key=score, reverse=True))[:top_n]
    return [
        Recommendation(rank=n, item=item, similarity=round(sim, 4),
                       explanation=explain(item, profile, request_text, constraints, lang))
        for n, (item, sim) in enumerate(ordered, start=1)
    ]
