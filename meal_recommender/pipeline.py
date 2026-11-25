"""End-to-end recommendation pipeline: constraints -> retrieval -> ranking -> final safety check."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from .constraints import filter_items, parse_request, violations
from .data import DEFAULT_MENU_PATH, load_menu
from .i18n import has_cjk, t
from .llm import LLMRanker
from .models import Recommendation, RecommendationResult, UserProfile
from .ranker import rank_with_rules
from .retriever import SemanticRetriever


class MealRecommender:
    def __init__(self, menu_path: Path | str = DEFAULT_MENU_PATH, api_key: Optional[str] = None,
                 model: Optional[str] = None, use_llm: Optional[bool] = None):
        self.items = load_menu(menu_path)
        self.retriever = SemanticRetriever(self.items)
        key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "")
        wants_llm = bool(key) if use_llm is None else use_llm
        self.llm: Optional[LLMRanker] = (
            LLMRanker(key, model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")) if wants_llm and key else None
        )

    @property
    def mode(self) -> str:
        return self.mode_label("en")

    def mode_label(self, lang: str = "en") -> str:
        return t("mode_llm", lang, model=self.llm.model) if self.llm else t("mode_rules", lang)

    def recommend(self, profile: UserProfile, request_text: str, top_n: int = 5, k: int = 15,
                  lang: str = "en") -> RecommendationResult:
        """Recommend dishes. `lang` sets the language of explanations and notes ("en" or "zh")."""
        notes: List[str] = []

        # 0. The menu and the embedding model are English: translate a Chinese request and dislikes first.
        search_text: Optional[str] = None
        if has_cjk(request_text) or any(has_cjk(d) for d in profile.dislikes):
            if self.llm:
                try:
                    search_text, dislikes = self.llm.translate(request_text, list(profile.dislikes))
                    profile = profile.model_copy(update={"dislikes": dislikes})
                    request_text = search_text
                except Exception as exc:
                    notes.append(t("note_translation_failed", lang, error=type(exc).__name__))
            else:
                notes.append(t("note_non_english_needs_llm", lang))

        # 1. Hard constraints: profile + limits written in the request ("under $12", "no mushrooms").
        constraints = parse_request(request_text)
        eligible = filter_items(self.items, profile, constraints)

        # 2. Semantic retrieval, searching only the eligible items.
        query = request_text.strip() or "a good meal"
        scored = self.retriever.search(query, k=k, allowed_ids=[i.item_id for i in eligible])

        # 3. Ranking + explanations: LLM if configured, otherwise rules. LLM problems fall back to rules.
        recs: List[Recommendation] = []
        source = "rules"
        if self.llm and scored:
            try:
                recs, rejected = self.llm.rank(scored, profile, request_text, constraints, top_n, lang)
                source = "llm"
                if rejected:
                    notes.append(t("note_discarded", lang, n=rejected))
            except Exception as exc:  # network, quota, malformed JSON ...
                notes.append(t("note_llm_unavailable", lang, error=type(exc).__name__))
                recs = []
        if len(recs) < min(top_n, len(scored)):
            chosen = {r.item.item_id for r in recs}
            filler = rank_with_rules([p for p in scored if p[0].item_id not in chosen],
                                     profile, request_text, constraints, top_n - len(recs), lang)
            if recs and filler:
                notes.append(t("note_filled", lang, n=len(filler)))
            for r in filler:
                recs.append(r.model_copy(update={"rank": len(recs) + 1}))

        # 4. Final safety net: nothing that breaks a hard constraint can leave the pipeline.
        safe = [r for r in recs if not violations(r.item, profile, constraints)]
        if len(safe) != len(recs):
            notes.append(t("note_removed", lang))
        safe = [r.model_copy(update={"rank": n}) for n, r in enumerate(safe, start=1)]

        if not eligible:
            notes.append(t("note_no_eligible", lang))

        return RecommendationResult(
            recommendations=safe,
            source=source,
            menu_size=len(self.items),
            eligible_count=len(eligible),
            retrieved_count=len(scored),
            constraints=constraints,
            notes=notes,
            search_text=search_text,
        )
