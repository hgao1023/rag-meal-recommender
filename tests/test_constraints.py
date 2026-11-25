import pytest

from meal_recommender.constraints import filter_items, parse_request, violations
from meal_recommender.data import load_menu
from meal_recommender.models import MenuItem, RequestConstraints, UserProfile

NO_LIMITS = RequestConstraints()


def item(**overrides):
    base = dict(item_id="X1", dish_name="Test Dish", restaurant="R", cuisine="C", price=10.0,
                description="tomato and basil", tags=[], allergens=[], dietary=[])
    base.update(overrides)
    return MenuItem(**base)


@pytest.mark.parametrize("text, expected", [
    ("Lunch under $15 please", 15.0),
    ("something below 12 dollars", 12.0),
    ("dinner, less than $9.50", 9.5),
    ("no more than 20 bucks", 20.0),
    ("a big dinner", None),
])
def test_parse_price(text, expected):
    assert parse_request(text).max_price == expected


def test_parse_excluded_terms_ignores_style_words():
    parsed = parse_request("high protein, no rice, not too greasy, without onions")
    assert parsed.excluded_terms == ["rice", "onions"]


def test_allergen_blocks_item():
    reasons = violations(item(allergens=["peanuts", "soy"]), UserProfile(allergies=["peanuts"]), NO_LIMITS)
    assert reasons and "peanuts" in reasons[0]


@pytest.mark.parametrize("diet, labels, allowed", [
    ("vegetarian", ["vegan"], True),
    ("vegetarian", ["pescatarian"], False),
    ("pescatarian", ["vegetarian"], True),
    ("vegan", ["vegetarian"], False),
    ("gluten_free", [], False),
])
def test_diet_rules(diet, labels, allowed):
    ok = not violations(item(dietary=labels), UserProfile(diet=diet), NO_LIMITS)
    assert ok is allowed


def test_request_price_is_stricter_than_budget():
    profile = UserProfile(budget=30)
    assert violations(item(price=14), profile, RequestConstraints(max_price=12))
    assert not violations(item(price=11), profile, RequestConstraints(max_price=12))


def test_dislike_matches_plural_and_singular():
    profile = UserProfile(dislikes=["mushrooms"])
    assert violations(item(description="roasted mushroom and thyme"), profile, NO_LIMITS)
    assert not violations(item(description="roasted squash"), profile, NO_LIMITS)


def test_menu_data_is_consistent():
    items = load_menu()
    assert len(items) == 150
    for i in items:
        if "vegan" in i.dietary:
            assert not set(i.allergens) & {"dairy", "eggs", "fish", "shellfish"}, i.item_id
        if "gluten_free" in i.dietary:
            assert "wheat" not in i.allergens, i.item_id


def test_filter_items_with_combined_profile():
    items = load_menu()
    profile = UserProfile(allergies=["dairy", "wheat"], diet="vegetarian", budget=12)
    kept = filter_items(items, profile, NO_LIMITS)
    assert kept
    assert all(i.price <= 12 and not {"dairy", "wheat"} & set(i.allergens) for i in kept)


def test_ingredient_groups_cover_specific_ingredients():
    from meal_recommender.data import load_menu

    menu = {i.item_id: i for i in load_menu()}
    no = lambda text: RequestConstraints(excluded_terms=[text])  # noqa: E731
    open_profile = UserProfile(budget=100)
    assert violations(menu["M001"], open_profile, no("cheese"))       # Margherita: mozzarella
    assert violations(menu["M011"], open_profile, no("pork"))         # Loaded Fries: bacon
    assert violations(menu["M085"], open_profile, no("meat"))         # Pepperoni Slice
    assert not violations(menu["M005"], open_profile, no("meat"))     # Caprese Salad
    assert violations(menu["M087"], open_profile, no("nuts"))         # cashew mozzarella, tree nut allergen


def test_zero_price_limit_is_respected():
    from meal_recommender.data import load_menu

    constraints = parse_request("max $0")
    assert constraints.max_price == 0
    assert all(violations(i, UserProfile(budget=100), constraints) for i in load_menu())
