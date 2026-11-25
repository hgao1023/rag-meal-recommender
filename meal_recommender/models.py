"""Data models shared by every stage of the pipeline."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

ALLERGENS = ["dairy", "eggs", "wheat", "soy", "peanuts", "tree_nuts", "fish", "shellfish", "sesame"]
DIETS = ["vegetarian", "vegan", "pescatarian", "gluten_free"]


class MenuItem(BaseModel):
    item_id: str
    dish_name: str
    restaurant: str
    cuisine: str
    price: float
    description: str
    tags: List[str] = Field(default_factory=list)
    allergens: List[str] = Field(default_factory=list)
    dietary: List[str] = Field(default_factory=list)

    def document(self) -> str:
        """Text that gets embedded for semantic search (tags are kept out on purpose)."""
        return f"{self.dish_name}. {self.cuisine} food from {self.restaurant}. {self.description}."


class UserProfile(BaseModel):
    allergies: List[str] = Field(default_factory=list)
    dislikes: List[str] = Field(default_factory=list)
    diet: Optional[str] = None
    budget: float = 20.0


class RequestConstraints(BaseModel):
    """Hard constraints parsed from the free-text request, e.g. "under $12, no mushrooms"."""
    max_price: Optional[float] = None
    excluded_terms: List[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    rank: int
    item: MenuItem
    similarity: float
    explanation: str


class RecommendationResult(BaseModel):
    recommendations: List[Recommendation]
    source: Literal["llm", "rules"]
    menu_size: int
    eligible_count: int
    retrieved_count: int
    constraints: RequestConstraints
    notes: List[str] = Field(default_factory=list)
    search_text: Optional[str] = None  # English text used for retrieval when the request was translated
