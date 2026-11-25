"""LLM re-ranking: the model may only choose among retrieved, constraint-checked items."""
from __future__ import annotations

import json
from typing import Dict, List, Tuple

from openai import OpenAI

from .models import MenuItem, Recommendation, RequestConstraints, UserProfile
from .retriever import ScoredItem

SYSTEM_PROMPT = """You recommend dishes for a food ordering assistant.

Rules:
- Choose ONLY from the candidate dishes provided. Refer to them by item_id. Never invent dishes.
- Every candidate already satisfies the user's allergies, diet and budget; do not reject any for those reasons.
- Rank by how well the dish matches the user's request (taste, style, occasion, healthiness).
- Prefer variety: avoid more than two picks from the same restaurant unless the request asks for one place.
- Each explanation: 1-2 sentences, specific to the request and grounded in the dish description, price and tags.
- Write each explanation in the language given in "explanation_language".
- Respond with JSON only: {"recommendations": [{"item_id": "...", "explanation": "..."}]}"""

TRANSLATE_PROMPT = """Translate a food ordering request and a list of disliked ingredients into English.
Keep prices, numbers and negations exactly ("15美元以内" -> "under $15", "不要米饭" -> "no rice").
Translate each disliked ingredient as a short common English ingredient name.
Respond with JSON only: {"request": "...", "dislikes": ["...", "..."]}"""

LANGUAGE_NAMES = {"en": "English", "zh": "Simplified Chinese"}


def _candidate_payload(item: MenuItem) -> Dict:
    return {
        "item_id": item.item_id,
        "dish": item.dish_name,
        "restaurant": item.restaurant,
        "cuisine": item.cuisine,
        "price": item.price,
        "description": item.description,
        "tags": item.tags,
    }


def build_messages(scored: List[ScoredItem], profile: UserProfile, request_text: str,
                   constraints: RequestConstraints, top_n: int, lang: str = "en") -> List[Dict[str, str]]:
    user = {
        "request": request_text,
        "profile": {
            "allergies": profile.allergies,
            "dislikes": profile.dislikes,
            "diet": profile.diet,
            "budget": min(profile.budget, constraints.max_price) if constraints.max_price is not None else profile.budget,
        },
        "return_top_n": top_n,
        "explanation_language": LANGUAGE_NAMES.get(lang, "English"),
        "candidates": [_candidate_payload(item) for item, _ in scored],
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user)},
    ]


def parse_response(content: str, scored: List[ScoredItem], top_n: int) -> Tuple[List[Recommendation], int]:
    """Validate the model output against the candidate list.

    Returns the accepted recommendations and the number of rejected item_ids
    (ids that were not in the candidate list, i.e. hallucinated, or duplicates).
    """
    by_id = {item.item_id: (item, sim) for item, sim in scored}
    data = json.loads(content)
    accepted: List[Recommendation] = []
    rejected = 0
    seen = set()
    for entry in data.get("recommendations", []):
        item_id = str(entry.get("item_id", "")).strip()
        explanation = str(entry.get("explanation", "")).strip()
        if item_id not in by_id or item_id in seen or not explanation:
            rejected += 1
            continue
        seen.add(item_id)
        item, sim = by_id[item_id]
        accepted.append(Recommendation(rank=len(accepted) + 1, item=item, similarity=round(sim, 4),
                                       explanation=explanation))
        if len(accepted) == top_n:
            break
    return accepted, rejected


class LLMRanker:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", timeout: float = 30.0):
        self.model = model
        self._client = OpenAI(api_key=api_key, timeout=timeout)

    def rank(self, scored: List[ScoredItem], profile: UserProfile, request_text: str,
             constraints: RequestConstraints, top_n: int = 5, lang: str = "en") -> Tuple[List[Recommendation], int]:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=build_messages(scored, profile, request_text, constraints, top_n, lang),
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        return parse_response(response.choices[0].message.content or "{}", scored, top_n)

    def translate(self, request_text: str, dislikes: List[str]) -> Tuple[str, List[str]]:
        """Translate a non-English request and dislikes to English, so retrieval and constraint checks work."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": TRANSLATE_PROMPT},
                      {"role": "user", "content": json.dumps({"request": request_text, "dislikes": dislikes},
                                                             ensure_ascii=False)}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return parse_translation(response.choices[0].message.content or "{}", request_text, dislikes)


def parse_translation(content: str, request_text: str, dislikes: List[str]) -> Tuple[str, List[str]]:
    data = json.loads(content)
    translated = str(data.get("request", "")).strip() or request_text
    out = [str(d).strip().lower() for d in data.get("dislikes", []) if str(d).strip()]
    if len(out) != len(dislikes):  # keep the user's dislikes if the model dropped or merged any
        out = [d for d in dislikes] + out
    return translated, out
