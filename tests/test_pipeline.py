import json
import random

import pytest

from meal_recommender import MealRecommender, UserProfile
from meal_recommender.constraints import parse_request, violations
from meal_recommender.llm import parse_response
from meal_recommender.models import ALLERGENS, DIETS


@pytest.fixture(scope="module")
def recommender():
    return MealRecommender(use_llm=False)


def test_semantic_search_finds_soups(recommender):
    top = recommender.retriever.search("a warm bowl of soup", k=5)
    assert sum("soup" in item.tags for item, _ in top) >= 4


def test_search_respects_allowed_ids(recommender):
    allowed = ["M004", "M046", "M150"]
    found = {item.item_id for item, _ in recommender.retriever.search("soup", k=10, allowed_ids=allowed)}
    assert found <= set(allowed)


def test_recommendations_never_break_constraints(recommender):
    rng = random.Random(0)
    requests = ["light lunch", "spicy noodles under $14", "comfort food, no cheese", "healthy breakfast",
                "something sweet", "high protein dinner"]
    for _ in range(60):
        profile = UserProfile(
            allergies=rng.sample(ALLERGENS, rng.randint(0, 3)),
            diet=rng.choice([None] + DIETS),
            budget=rng.choice([10, 15, 20, 30]),
        )
        text = rng.choice(requests)
        result = recommender.recommend(profile, text)
        for rec in result.recommendations:
            assert not violations(rec.item, profile, parse_request(text)), (profile, text, rec.item.item_id)


def test_impossible_profile_returns_empty_with_note(recommender):
    result = recommender.recommend(UserProfile(diet="vegan", budget=3), "dinner")
    assert result.recommendations == []
    assert any("No menu item" in n for n in result.notes)


def test_llm_output_is_validated_against_candidates(recommender):
    scored = recommender.retriever.search("soup", k=5)
    real_ids = [item.item_id for item, _ in scored]
    content = json.dumps({"recommendations": [
        {"item_id": "M999", "explanation": "invented dish"},          # not a candidate
        {"item_id": real_ids[1], "explanation": "good match"},
        {"item_id": real_ids[1], "explanation": "duplicate"},         # duplicate
        {"item_id": real_ids[0], "explanation": "also good"},
    ]})
    recs, rejected = parse_response(content, scored, top_n=5)
    assert [r.item.item_id for r in recs] == [real_ids[1], real_ids[0]]
    assert [r.rank for r in recs] == [1, 2]
    assert rejected == 2


def test_llm_failure_falls_back_to_rules(recommender):
    class BrokenLLM:
        model = "broken"

        def rank(self, *args, **kwargs):
            raise TimeoutError("simulated outage")

    recommender.llm = BrokenLLM()
    try:
        result = recommender.recommend(UserProfile(budget=20), "light lunch")
    finally:
        recommender.llm = None
    assert result.source == "rules"
    assert len(result.recommendations) == 5
    assert any("LLM unavailable" in n for n in result.notes)


class FakeTranslatingLLM:
    model = "fake"

    def translate(self, request_text, dislikes):
        return "spicy noodles under $14", ["mushroom"]

    def rank(self, scored, profile, request_text, constraints, top_n, lang="en"):
        raise RuntimeError("use rules")


def test_chinese_request_is_translated_before_retrieval(recommender):
    recommender.llm = FakeTranslatingLLM()
    try:
        result = recommender.recommend(UserProfile(dislikes=["蘑菇"]), "来点辣的面，14 美元以内", lang="zh")
    finally:
        recommender.llm = None
    assert result.search_text == "spicy noodles under $14"
    assert result.constraints.max_price == 14
    assert all(r.item.price <= 14 and "mushroom" not in r.item.description.lower() for r in result.recommendations)
    assert any("大模型暂时不可用" in n for n in result.notes)  # notes follow the UI language


def test_chinese_request_without_llm_warns(recommender):
    result = recommender.recommend(UserProfile(), "来点辣的", lang="zh")
    assert any("API 密钥" in n for n in result.notes)
    assert len(result.recommendations) == 5
