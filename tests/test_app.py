"""Tests of the Streamlit app itself, run headless with Streamlit's AppTest."""
import pytest
from streamlit.testing.v1 import AppTest

from meal_recommender.models import ALLERGENS


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")  # rule-based mode: no network calls in tests
    at = AppTest.from_file("../app.py", default_timeout=120)
    at.run()
    assert not at.exception
    return at


def get_button(at, label):
    return next(b for b in at.button if b.label == label)


def test_default_city_gives_zip_and_enabled_button(app):
    city = next(s for s in app.sidebar.selectbox if s.label == "City")
    assert city.value == "San Jose, CA"
    assert not get_button(app, "Get recommendations").disabled


def test_zip_mode_requires_a_valid_zip(app):
    app.sidebar.radio[1].set_value("free").run()
    assert get_button(app, "Get recommendations").disabled
    assert any("Enter a ZIP code" in w.value for w in app.sidebar.warning)

    app.sidebar.text_input[0].set_value("95053!!").run()
    assert get_button(app, "Get recommendations").disabled
    assert any("exactly 5 digits" in e.value for e in app.sidebar.error)

    app.sidebar.text_input[0].set_value("99999").run()
    assert get_button(app, "Get recommendations").disabled
    assert any("does not exist" in e.value for e in app.sidebar.error)

    app.sidebar.text_input[0].set_value("500 El Camino Real, Santa Clara, CA 95053").run()
    assert not get_button(app, "Get recommendations").disabled
    assert any("95053" in s.value and "Santa Clara" in s.value for s in app.sidebar.success)


def test_recommendations_render_with_zip_links(app):
    app.sidebar.radio[1].set_value("free").run()
    app.sidebar.text_input[0].set_value("94103").run()
    app.sidebar.multiselect[0].set_value(["peanuts", "shellfish"]).run()
    get_button(app, "Get recommendations").click().run()
    assert not app.exception
    assert len(app.subheader) == 5
    assert any("ZIP 94103" in c.value for c in app.caption)


def test_example_button_fills_the_request(app):
    get_button(app, "Warm soup for a cold rainy evening").click().run()
    assert app.text_area[0].value == "Warm soup for a cold rainy evening"


def test_impossible_profile_shows_a_message_instead_of_failing(app):
    app.sidebar.multiselect[0].set_value(ALLERGENS).run()
    app.sidebar.slider[0].set_value(5).run()
    app.sidebar.text_input[0].set_value("collard greens").run()  # the only $5 dish with no tracked allergen
    get_button(app, "Get recommendations").click().run()
    assert not app.exception
    assert len(app.subheader) == 0
    assert any("No menu item satisfies" in i.value for i in app.info)


def test_empty_request_still_recommends(app):
    app.text_area[0].set_value("").run()
    get_button(app, "Get recommendations").click().run()
    assert not app.exception
    assert len(app.subheader) == 5


def test_chinese_interface(app):
    app.sidebar.radio[0].set_value("zh").run()
    assert app.title[0].value.endswith("今天吃什么？")
    assert app.text_area[0].value == "今天午饭，15 美元以内，清淡一点不要太油"
    get_button(app, "获取推荐").click().run()
    assert not app.exception
    assert len(app.subheader) == 5
    assert any("API 密钥" in i.value for i in app.info)  # no key in tests: Chinese request warning

    app.sidebar.radio[0].set_value("en").run()
    assert app.text_area[0].value == "Lunch today, under $15, something light and not too greasy"
