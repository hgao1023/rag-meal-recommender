"""English / Chinese text for the app, notes and rule-based explanations.

Menu data (dish names, restaurants, descriptions) stays in English.
"""
from __future__ import annotations

import re
from typing import Dict

LANGUAGES = {"en": "English", "zh": "中文"}

LABELS: Dict[str, Dict[str, str]] = {
    "en": {},  # English labels are derived from the key ("high_protein" -> "High Protein")
    "zh": {
        # allergens
        "dairy": "乳制品", "eggs": "鸡蛋", "wheat": "小麦", "soy": "大豆", "peanuts": "花生",
        "tree_nuts": "坚果", "fish": "鱼类", "shellfish": "贝类/甲壳类", "sesame": "芝麻",
        # diets
        "vegetarian": "素食（蛋奶素）", "vegan": "纯素", "pescatarian": "鱼素", "gluten_free": "无麸质",
        # tags
        "breakfast": "早餐", "comfort": "暖心家常", "dessert": "甜点", "fried": "油炸", "grilled": "烧烤",
        "healthy": "健康", "high_protein": "高蛋白", "light": "清淡", "low_carb": "低碳水", "noodles": "面食",
        "quick": "快手", "rice": "米饭", "salad": "沙拉", "sandwich": "三明治/卷饼", "soup": "汤",
        "spicy": "辣", "sweet": "甜",
    },
}

TEXT: Dict[str, Dict[str, str]] = {
    "en": {
        "page_title": "What should I eat?",
        "intro": "Describe what you feel like eating. Allergies, diet and budget are always respected.",
        "profile": "Your profile",
        "location": "Your location",
        "pick_city": "Pick a city",
        "enter_location": "ZIP, address or place",
        "city": "City",
        "location_input": "ZIP code, address or place name",
        "location_placeholder": "e.g. 95053, an address, or NYU",
        "location_help": "A ZIP code is required. Type a 5-digit ZIP code, an address that ends with one, "
                         "or a place name (looked up on OpenStreetMap). Press Enter to check it.",
        "location_ok": "Using ZIP **{zip}** ({label})",
        "loc_empty": "Enter a ZIP code, an address or a place, then press Enter.",
        "loc_format": "A ZIP code is exactly 5 digits (e.g. 95053) or ZIP+4 (95053-1234), with no other symbols.",
        "loc_unknown_zip": "This ZIP code does not exist in the US ZIP code list.",
        "loc_not_found": "Could not find this place in the US. Try a ZIP code or a fuller name.",
        "loc_lookup_failed": "The place lookup is not available right now. Please enter a ZIP code instead.",
        "allergies": "Allergies",
        "choose": "Choose allergens",
        "diet": "Diet",
        "no_diet": "No restriction",
        "dislikes": "Ingredients you dislike",
        "dislikes_placeholder": "e.g. mushroom, cilantro",
        "budget": "Budget per meal ($)",
        "menu_caption": "Menu: {items} dishes from {restaurants} restaurants (sample data)",
        "mode_caption": "Ranking mode: **{mode}**",
        "mode_llm": "LLM ({model})",
        "mode_rules": "rules (no API key)",
        "theme_hint": "Light / dark theme: ⋮ menu (top right) → Settings.",
        "request": "Your request",
        "get": "Get recommendations",
        "need_location": "Enter a valid location in the sidebar first",
        "spinner": "Finding dishes...",
        "m_menu": "Menu items",
        "m_eligible": "Pass your constraints",
        "m_retrieved": "Retrieved by meaning",
        "m_recommended": "Recommended",
        "understood": "Understood from your request: ",
        "max_price": "max price ${price:.0f}",
        "excluded": "excluded: {terms}",
        "translated": "searched as: \"{text}\"",
        "tags": "Tags",
        "contains": "Contains",
        "no_allergens": "None of the 9 tracked allergens",
        "find_nearby": "📍 Find nearby",
        "yelp": "⭐ Search on Yelp",
        "maps_help": "Search Google Maps for this dish near {zip}",
        "yelp_help": "Search Yelp for this dish near {zip}",
        "footer": "Ranked and explained by {source}. Restaurants in the menu are sample data, so the buttons "
                  "search for the dish near ZIP {zip} on Google Maps and Yelp.",
        "by_llm": "the LLM",
        "by_rules": "rule-based ranking",
        "how": "How it works",
        "how_body": """
1. **Hard constraints** remove every dish that conflicts with your allergies, diet, budget, disliked ingredients,
   or limits written in the request ("under $12", "no rice"). This is done in code, not left to the model.
2. **Semantic retrieval** embeds your request and finds the closest remaining dishes in a ChromaDB vector index.
3. **Ranking and explanation**: an LLM picks the best matches *from the retrieved list only* and explains each choice.
   Without an API key, a rule-based ranker and template explanations are used instead.
4. **Final check**: every recommendation is re-validated against the hard constraints before it is shown.
""",
        # pipeline notes
        "note_discarded": "Discarded {n} LLM pick(s) that were not in the candidate list.",
        "note_llm_unavailable": "LLM unavailable ({error}); used rule-based ranking.",
        "note_filled": "Filled {n} slot(s) with rule-based picks.",
        "note_removed": "Removed item(s) that failed the final constraint check.",
        "note_no_eligible": "No menu item satisfies all constraints. Try relaxing the budget or diet.",
        "note_translation_failed": "Could not translate the request ({error}); searched with the original text.",
        "note_non_english_needs_llm": "Requests and dislikes in Chinese need an OpenAI API key to be translated; "
                                      "without it, results may not match and Chinese dislikes are not checked.",
    },
    "zh": {
        "page_title": "今天吃什么？",
        "intro": "说说你想吃什么。过敏原、饮食限制和预算一定会被遵守。",
        "profile": "个人设置",
        "location": "你的位置",
        "pick_city": "选择城市",
        "enter_location": "邮编、地址或地名",
        "city": "城市",
        "location_input": "邮编、地址或地名",
        "location_placeholder": "例如 95053、一个地址，或 NYU",
        "location_help": "必须有邮编。可以输入 5 位邮编、以邮编结尾的地址，或地名（通过 OpenStreetMap 查询）。"
                         "输入后按回车检查。",
        "location_ok": "使用邮编 **{zip}**（{label}）",
        "loc_empty": "请输入邮编、地址或地名，然后按回车。",
        "loc_format": "邮编必须正好是 5 位数字（例如 95053），或 5+4 位（95053-1234），不能有其他符号。",
        "loc_unknown_zip": "美国邮编库里没有这个邮编。",
        "loc_not_found": "在美国没有找到这个地方。请改填邮编，或写更完整的名称。",
        "loc_lookup_failed": "地名查询暂时不可用，请直接填写邮编。",
        "allergies": "过敏原",
        "choose": "选择过敏原",
        "diet": "饮食限制",
        "no_diet": "无限制",
        "dislikes": "不喜欢的食材",
        "dislikes_placeholder": "例如 mushroom, cilantro",
        "budget": "每餐预算（美元）",
        "menu_caption": "菜单：{restaurants} 家餐厅的 {items} 道菜（示例数据）",
        "mode_caption": "排序方式：**{mode}**",
        "mode_llm": "大模型（{model}）",
        "mode_rules": "规则（未设置 API 密钥）",
        "theme_hint": "亮色 / 暗色模式：右上角 ⋮ 菜单 → Settings。",
        "request": "你的需求",
        "get": "获取推荐",
        "need_location": "请先在侧边栏填写有效的位置",
        "spinner": "正在找菜...",
        "m_menu": "菜单菜品",
        "m_eligible": "符合你的限制",
        "m_retrieved": "语义检索到",
        "m_recommended": "最终推荐",
        "understood": "从你的需求中识别到：",
        "max_price": "最高价格 ${price:.0f}",
        "excluded": "排除：{terms}",
        "translated": "检索用的英文：\"{text}\"",
        "tags": "标签",
        "contains": "含有",
        "no_allergens": "不含 9 种常见过敏原",
        "find_nearby": "📍 附近找找",
        "yelp": "⭐ 在 Yelp 搜索",
        "maps_help": "在 Google 地图上搜索邮编 {zip} 附近的这道菜",
        "yelp_help": "在 Yelp 上搜索邮编 {zip} 附近的这道菜",
        "footer": "由{source}排序并说明理由。菜单里的餐厅是示例数据，所以按钮会在 Google 地图和 Yelp 上"
                  "搜索邮编 {zip} 附近的这道菜。",
        "by_llm": "大模型",
        "by_rules": "规则",
        "how": "工作原理",
        "how_body": """
1. **硬性限制**：先排除所有和你的过敏原、饮食限制、预算、不喜欢的食材，或需求里写明的限制（"under $12"、"不要米饭"）冲突的菜。
   这一步由代码完成，不交给模型判断。
2. **语义检索**：把你的需求转成向量，在 ChromaDB 向量库中找出含义最接近的菜。
3. **排序和解释**：大模型*只能从检索到的菜里*挑选，并说明理由。没有 API 密钥时，改用规则排序和模板说明。
4. **最终检查**：每道推荐的菜在显示前都会再按硬性限制检查一遍。
""",
        "note_discarded": "丢弃了 {n} 个不在候选列表里的大模型推荐。",
        "note_llm_unavailable": "大模型暂时不可用（{error}），已改用规则排序。",
        "note_filled": "有 {n} 个名额由规则排序补上。",
        "note_removed": "移除了未通过最终检查的菜。",
        "note_no_eligible": "没有菜能同时满足所有限制。可以试试放宽预算或饮食限制。",
        "note_translation_failed": "需求翻译失败（{error}），已用原文检索。",
        "note_non_english_needs_llm": "中文需求和中文的忌口需要 OpenAI API 密钥才能翻译；没有密钥时，推荐可能不准，"
                                      "中文写的忌口也无法检查。",
    },
}

EXAMPLES = {
    "en": [
        "Lunch today, under $15, something light and not too greasy",
        "Warm soup for a cold rainy evening",
        "High protein dinner after the gym, no rice",
        "Something spicy and quick",
    ],
    "zh": [
        "今天午饭，15 美元以内，清淡一点不要太油",
        "下雨天想喝一碗热汤",
        "健身后的高蛋白晚餐，不要米饭",
        "来点辣的，快一点",
    ],
}

_CJK = re.compile(r"[㐀-鿿豈-﫿]")


def has_cjk(text: str) -> bool:
    return bool(_CJK.search(text or ""))


def t(key: str, lang: str = "en", **kwargs) -> str:
    text = TEXT.get(lang, TEXT["en"]).get(key) or TEXT["en"][key]
    return text.format(**kwargs) if kwargs else text


def label(value: str, lang: str = "en") -> str:
    return LABELS.get(lang, {}).get(value) or value.replace("_", " ").title()
