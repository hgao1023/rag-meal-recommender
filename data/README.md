# Menu data

`menu_items.csv` is a **synthetic sample menu** written for this demo: 150 dishes from 25 fictional restaurants across 20 cuisines. Restaurant names are invented and do not refer to real businesses. Prices are typical San Francisco Bay Area prices; allergen and diet labels were assigned by hand from each dish's ingredients.

| Column | Meaning |
| --- | --- |
| `item_id` | Unique id (M001-M150) |
| `dish_name`, `restaurant`, `cuisine` | What and where |
| `price` | Price in USD |
| `description` | Short ingredient description (this text is embedded for semantic search) |
| `tags` | Style tags separated by `;` (light, spicy, soup, high_protein, ...). Used for explanations and as evaluation labels, **not** embedded |
| `allergens` | Any of: dairy, eggs, wheat, soy, peanuts, tree_nuts, fish, shellfish, sesame |
| `dietary` | Any of: vegetarian, vegan, pescatarian, gluten_free |

`load_menu()` rejects unknown labels and duplicate ids, and `tests/test_constraints.py` checks label consistency (for example, no item labelled vegan contains dairy or eggs).

The labels are suitable for a demo, not for real allergy decisions.
