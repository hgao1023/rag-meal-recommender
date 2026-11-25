"""Streamlit front end for the meal recommender."""
from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from meal_recommender import MealRecommender, UserProfile
from meal_recommender.i18n import EXAMPLES, LANGUAGES, label, t
from meal_recommender.links import order_links
from meal_recommender.location import CITY_ZIPS, DEFAULT_CITY, LocationResult, resolve_location
from meal_recommender.models import ALLERGENS, DIETS

load_dotenv()

st.set_page_config(page_title="Meal Recommender", page_icon="🍽️", layout="wide")


@st.cache_resource(show_spinner="Indexing the menu...")
def get_recommender() -> MealRecommender:
    return MealRecommender()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def lookup_location(text: str) -> LocationResult:
    return resolve_location(text)


recommender = get_recommender()

# ---------------------------------------------------------------- sidebar: language, location, profile
with st.sidebar:
    lang = st.radio("Language / 语言", list(LANGUAGES), format_func=LANGUAGES.get, horizontal=True, key="lang")
    st.header(t("profile", lang), anchor=False)

    st.markdown(f"**{t('location', lang)}**")
    how = st.radio("location_mode", ["city", "free"], horizontal=True, label_visibility="collapsed",
                   format_func=lambda m: t("pick_city" if m == "city" else "enter_location", lang))
    zip_code = None
    if how == "city":
        cities = list(CITY_ZIPS)
        city = st.selectbox(t("city", lang), cities, index=cities.index(DEFAULT_CITY),
                            format_func=lambda c: f"{c} ({CITY_ZIPS[c]})")
        zip_code = CITY_ZIPS[city]
    else:
        typed = st.text_input(t("location_input", lang), placeholder=t("location_placeholder", lang),
                              help=t("location_help", lang))
        loc = lookup_location(typed.strip())
        if loc.ok:
            zip_code = loc.zip_code
            st.success(t("location_ok", lang, zip=loc.zip_code, label=loc.label), icon="📍")
        elif loc.error == "empty":
            st.warning(t("loc_empty", lang))
        else:
            st.error(t(f"loc_{loc.error}", lang))

    allergies = st.multiselect(t("allergies", lang), ALLERGENS, format_func=lambda a: label(a, lang),
                               placeholder=t("choose", lang))
    diet = st.selectbox(t("diet", lang), [None] + DIETS,
                        format_func=lambda d: t("no_diet", lang) if d is None else label(d, lang))
    dislikes_text = st.text_input(t("dislikes", lang), placeholder=t("dislikes_placeholder", lang))
    budget = st.slider(t("budget", lang), 5, 40, 20)
    st.divider()
    st.caption(t("menu_caption", lang, items=len(recommender.items),
                 restaurants=len({i.restaurant for i in recommender.items})))
    st.caption(t("mode_caption", lang, mode=recommender.mode_label(lang)))
    st.caption(t("theme_hint", lang))

profile = UserProfile(
    allergies=allergies,
    dislikes=[d.strip() for d in dislikes_text.replace("，", ",").split(",") if d.strip()],
    diet=diet,
    budget=float(budget),
)

# ---------------------------------------------------------------- main: request
st.title("🍽️ " + t("page_title", lang), anchor=False)
st.write(t("intro", lang))

examples = EXAMPLES[lang]
if st.session_state.get("request_lang") != lang:  # switch the default request with the language
    if st.session_state.get("request") in (None, *EXAMPLES["en"], *EXAMPLES["zh"]):
        st.session_state.request = examples[0]
    st.session_state.request_lang = lang


def use_example(text: str) -> None:
    st.session_state.request = text


cols = st.columns(len(examples))
for col, example in zip(cols, examples):
    col.button(example, use_container_width=True, on_click=use_example, args=(example,))

request = st.text_area(t("request", lang), key="request", height=80)

if st.button(t("get", lang), type="primary", disabled=not zip_code,
             help=None if zip_code else t("need_location", lang)):
    with st.spinner(t("spinner", lang)):
        result = recommender.recommend(profile, request, lang=lang)

    c = result.constraints
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(t("m_menu", lang), result.menu_size)
    m2.metric(t("m_eligible", lang), result.eligible_count)
    m3.metric(t("m_retrieved", lang), result.retrieved_count)
    m4.metric(t("m_recommended", lang), len(result.recommendations))
    understood = []
    if result.search_text:
        understood.append(t("translated", lang, text=result.search_text))
    if c.max_price is not None:
        understood.append(t("max_price", lang, price=c.max_price))
    if c.excluded_terms:
        understood.append(t("excluded", lang, terms=", ".join(c.excluded_terms)))
    if understood:
        st.caption(t("understood", lang) + "; ".join(understood))
    for note in result.notes:
        st.info(note)

    for rec in result.recommendations:
        item = rec.item
        with st.container(border=True):
            left, right = st.columns([4, 1])
            left.subheader(f"{rec.rank}. {item.dish_name}", anchor=False)
            left.write(f"**{item.restaurant}** · {item.cuisine} · ${item.price:.2f}")
            left.write(rec.explanation)
            left.caption(item.description)
            links = order_links(item, zip_code)
            b1, b2, _ = left.columns([1, 1, 1.5])
            b1.link_button(t("find_nearby", lang), links["Google Maps"], use_container_width=True,
                           help=t("maps_help", lang, zip=zip_code))
            b2.link_button(t("yelp", lang), links["Yelp"], use_container_width=True,
                           help=t("yelp_help", lang, zip=zip_code))
            right.write(f"**{t('tags', lang)}**")
            right.write(", ".join(label(x, lang) for x in item.tags) or "-")
            right.write(f"**{t('contains', lang)}**")
            right.write(", ".join(label(a, lang) for a in item.allergens) or t("no_allergens", lang))

    source = t("by_llm" if result.source == "llm" else "by_rules", lang)
    st.caption(t("footer", lang, source=source, zip=zip_code))

with st.expander(t("how", lang)):
    st.markdown(t("how_body", lang))
