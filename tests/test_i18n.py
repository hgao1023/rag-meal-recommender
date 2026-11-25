from meal_recommender.data import load_menu
from meal_recommender.i18n import EXAMPLES, LABELS, TEXT, has_cjk, label, t
from meal_recommender.models import ALLERGENS, DIETS, RequestConstraints, UserProfile
from meal_recommender.ranker import explain


def test_every_text_has_both_languages():
    assert set(TEXT["en"]) == set(TEXT["zh"])
    assert len(EXAMPLES["en"]) == len(EXAMPLES["zh"])


def test_every_label_is_translated():
    tags = {tag for item in load_menu() for tag in item.tags}
    for value in [*ALLERGENS, *DIETS, *tags]:
        assert value in LABELS["zh"], value
    assert label("high_protein", "en") == "High Protein"


def test_format_and_cjk_detection():
    assert t("note_discarded", "zh", n=2) == "丢弃了 2 个不在候选列表里的大模型推荐。"
    assert has_cjk("来点辣的") and not has_cjk("something spicy")


def test_rule_explanation_in_chinese():
    item = next(i for i in load_menu() if "light" in i.tags and "fried" not in i.tags)
    text = explain(item, UserProfile(allergies=["peanuts"], diet=None, budget=15), "light lunch, not greasy",
                   RequestConstraints(), lang="zh")
    assert "清淡" in text and "不是油炸的" in text and "花生" in text and text.endswith("。")
