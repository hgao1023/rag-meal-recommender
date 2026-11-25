from urllib.parse import parse_qs, urlparse

from meal_recommender.data import load_menu
from meal_recommender.links import order_links
from meal_recommender.location import DEFAULT_ZIP


def test_order_links_search_the_dish_near_the_zip():
    item = next(i for i in load_menu() if i.item_id == "M055")  # Pho Bo
    links = order_links(item, "95050")
    maps = parse_qs(urlparse(links["Google Maps"]).query)
    yelp = parse_qs(urlparse(links["Yelp"]).query)
    assert maps["query"] == ["Pho Bo Vietnamese near 95050"]
    assert yelp["find_desc"] == ["Pho Bo Vietnamese"] and yelp["find_loc"] == ["95050"]
    assert parse_qs(urlparse(order_links(item, "oops")["Yelp"]).query)["find_loc"] == [DEFAULT_ZIP]
