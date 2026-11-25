"""Load and validate the menu dataset."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List

from .models import ALLERGENS, DIETS, MenuItem

DEFAULT_MENU_PATH = Path(__file__).resolve().parent.parent / "data" / "menu_items.csv"


def _split(value: str) -> List[str]:
    return [part.strip() for part in value.split(";") if part.strip()]


def load_menu(path: Path | str = DEFAULT_MENU_PATH) -> List[MenuItem]:
    items: List[MenuItem] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            item = MenuItem(
                item_id=row["item_id"],
                dish_name=row["dish_name"],
                restaurant=row["restaurant"],
                cuisine=row["cuisine"],
                price=float(row["price"]),
                description=row["description"],
                tags=_split(row["tags"]),
                allergens=_split(row["allergens"]),
                dietary=_split(row["dietary"]),
            )
            unknown = (set(item.allergens) - set(ALLERGENS)) | (set(item.dietary) - set(DIETS))
            if unknown:
                raise ValueError(f"{item.item_id}: unknown labels {sorted(unknown)}")
            items.append(item)

    ids = [i.item_id for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate item_id values in menu data")
    return items
