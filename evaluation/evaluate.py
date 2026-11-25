"""Offline evaluation of the meal recommender.

1. Safety: how often do recommendations violate a hard constraint (allergy, diet, budget, dislike)?
   Compared with retrieval alone, without the constraint filter.
2. Relevance: Precision@5 on 30 labelled natural-language requests, comparing
   random picks, keyword matching, semantic retrieval, and the full pipeline.
3. (optional, --llm) The same relevance test with LLM re-ranking, plus how many LLM picks
   had to be discarded for not being in the candidate list.

Usage (from the repository root):
    python -m evaluation.evaluate          # no API key needed
    python -m evaluation.evaluate --llm    # also evaluates LLM re-ranking (needs OPENAI_API_KEY)
"""
from __future__ import annotations

import argparse
import random
import statistics
import time
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
from dotenv import load_dotenv

from meal_recommender import MealRecommender, UserProfile
from meal_recommender.constraints import parse_request, violations
from meal_recommender.models import ALLERGENS, DIETS, MenuItem, RequestConstraints
from meal_recommender.retriever import KeywordRetriever

ROOT = Path(__file__).resolve().parent.parent
OUT_MD = ROOT / "evaluation" / "results.md"
OUT_PNG = ROOT / "images" / "evaluation_relevance.png"
SEED = 7
K = 5

# Each request is labelled with what a relevant dish must have: a tag ("tag:soup") or a cuisine ("cuisine:Thai").
# Requests are phrased the way people talk, and mostly avoid the literal tag word.
LABELLED_REQUESTS: List[tuple[str, str]] = [
    ("Something warm and brothy for a cold rainy day", "tag:soup"),
    ("I'm sick and want a comforting bowl of broth", "tag:soup"),
    ("A bowl of noodles in hot broth", "tag:noodles"),
    ("Slurpable noodles for lunch", "tag:noodles"),
    ("I want something with a lot of chili heat", "tag:spicy"),
    ("Food that will make my mouth burn", "tag:spicy"),
    ("A crisp bowl of fresh greens", "tag:salad"),
    ("Leafy greens with a vinaigrette", "tag:salad"),
    ("Something between two slices of bread", "tag:sandwich"),
    ("A wrap or sandwich I can eat at my desk", "tag:sandwich"),
    ("First meal of the day, maybe eggs or pancakes", "tag:breakfast"),
    ("Brunch on a Sunday morning", "tag:breakfast"),
    ("Something sugary to finish dinner", "tag:sweet"),
    ("I have a sweet tooth tonight", "tag:sweet"),
    ("Deep-fried and crispy, I don't care about calories", "tag:fried"),
    ("Something battered and crunchy", "tag:fried"),
    ("Lots of meat or fish to hit my protein goal after lifting", "tag:high_protein"),
    ("A steak or chicken dinner to refuel", "tag:high_protein"),
    ("Charred off the grill", "tag:grilled"),
    ("Smoky barbecue", "tag:grilled"),
    ("Sushi or ramen", "cuisine:Japanese"),
    ("Thai food with coconut and lemongrass", "cuisine:Thai"),
    ("Curry with naan and basmati rice", "cuisine:Indian"),
    ("Tacos or a burrito", "cuisine:Mexican"),
    ("Pasta or pizza like in Italy", "cuisine:Italian"),
    ("Kimchi and gochujang", "cuisine:Korean"),
    ("Pho or a banh mi", "cuisine:Vietnamese"),
    ("Something from the ocean, like shellfish", "cuisine:Seafood"),
    ("A raw fish rice bowl from Hawaii", "cuisine:Hawaiian"),
    ("Dim sum or a stir-fry from a Chinese place", "cuisine:Chinese"),
]


def is_relevant(item: MenuItem, label: str) -> bool:
    kind, value = label.split(":", 1)
    return value in item.tags if kind == "tag" else item.cuisine == value


def precision_at_k(items: List[MenuItem], label: str) -> float:
    return sum(is_relevant(i, label) for i in items[:K]) / K


def evaluate_relevance(rec: MealRecommender, methods: Dict[str, Callable[[str], List[MenuItem]]]) -> Dict[str, float]:
    scores = {}
    for name, fn in methods.items():
        scores[name] = statistics.mean(precision_at_k(fn(text), label) for text, label in LABELLED_REQUESTS)
    return scores


def evaluate_safety(rec: MealRecommender, n: int = 300) -> Dict[str, float]:
    rng = random.Random(SEED)
    texts = [t for t, _ in LABELLED_REQUESTS]
    pipeline_bad = retrieval_bad = total_pipeline = total_retrieval = 0
    latencies = []
    for _ in range(n):
        profile = UserProfile(
            allergies=rng.sample(ALLERGENS, rng.randint(1, 3)),
            diet=rng.choice([None, None] + DIETS),
            budget=float(rng.choice([10, 12, 15, 20, 25])),
        )
        text = rng.choice(texts)
        constraints = parse_request(text)

        start = time.perf_counter()
        result = rec.recommend(profile, text)
        latencies.append(time.perf_counter() - start)
        total_pipeline += len(result.recommendations)
        pipeline_bad += sum(bool(violations(r.item, profile, constraints)) for r in result.recommendations)

        retrieved = [i for i, _ in rec.retriever.search(text, k=K)]  # no constraint filter
        total_retrieval += len(retrieved)
        retrieval_bad += sum(bool(violations(i, profile, constraints)) for i in retrieved)
    return {
        "cases": n,
        "pipeline_violation_rate": pipeline_bad / max(total_pipeline, 1),
        "pipeline_violations": pipeline_bad,
        "pipeline_items": total_pipeline,
        "retrieval_only_violation_rate": retrieval_bad / max(total_retrieval, 1),
        "retrieval_only_violations": retrieval_bad,
        "retrieval_only_items": total_retrieval,
        "median_latency_ms": statistics.median(latencies) * 1000,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="also evaluate LLM re-ranking (needs OPENAI_API_KEY)")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")

    rules = MealRecommender(use_llm=False)
    keyword = KeywordRetriever(rules.items)
    open_profile = UserProfile(budget=100)
    rng = random.Random(SEED)

    methods: Dict[str, Callable[[str], List[MenuItem]]] = {
        "Random": lambda t: rng.sample(rules.items, K),
        "Keyword match": lambda t: [i for i, _ in keyword.search(t, k=K)],
        "Semantic retrieval": lambda t: [i for i, _ in rules.retriever.search(t, k=K)],
        "Full pipeline (rules)": lambda t: [r.item for r in rules.recommend(open_profile, t).recommendations],
    }

    llm_rejected = llm_total = 0
    if args.llm:
        llm_rec = MealRecommender(use_llm=True)
        if not llm_rec.llm:
            raise SystemExit("OPENAI_API_KEY is not set; cannot run --llm.")

        def run_llm(text: str) -> List[MenuItem]:
            nonlocal llm_rejected, llm_total
            result = llm_rec.recommend(open_profile, text)
            llm_total += 1
            llm_rejected += sum(int(n.split()[1]) for n in result.notes if n.startswith("Discarded"))
            return [r.item for r in result.recommendations]

        methods[f"Full pipeline ({llm_rec.llm.model})"] = run_llm

    relevance = evaluate_relevance(rules, methods)
    safety = evaluate_safety(rules)

    # chart
    names = list(relevance)
    values = [relevance[n] for n in names]
    fig, ax = plt.subplots(figsize=(7.5, 0.6 * len(names) + 1.2))
    best = max(values)
    colors = ["#E4572E" if v == best else "#9AA5B1" for v in values[::-1]]  # highlight the best method
    bars = ax.barh(names[::-1], values[::-1], color=colors)
    for bar, v in zip(bars, values[::-1]):
        ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2, f"{v:.2f}", va="center")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel(f"Precision@{K} on {len(LABELLED_REQUESTS)} labelled requests")
    ax.set_title("How often the top 5 dishes match the request")
    plt.tight_layout()
    OUT_PNG.parent.mkdir(exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150)

    lines = [
        "# Evaluation results",
        "",
        "Generated by `python -m evaluation.evaluate" + (" --llm" if args.llm else "") + "`.",
        "",
        f"## Relevance (Precision@{K}, {len(LABELLED_REQUESTS)} labelled requests, no constraints)",
        "",
        "| Method | Precision@5 |",
        "| --- | --- |",
        *[f"| {n} | {v:.2f} |" for n, v in relevance.items()],
        "",
        f"## Safety ({safety['cases']} random profiles with 1-3 allergies, optional diet, budget $10-25)",
        "",
        "| Method | Items checked | Items violating a constraint | Violation rate |",
        "| --- | --- | --- | --- |",
        f"| Semantic retrieval only (no filter) | {safety['retrieval_only_items']} | "
        f"{safety['retrieval_only_violations']} | {safety['retrieval_only_violation_rate']:.1%} |",
        f"| Full pipeline | {safety['pipeline_items']} | {safety['pipeline_violations']} | "
        f"{safety['pipeline_violation_rate']:.1%} |",
        "",
        f"Median latency per request (rules mode, local embeddings): {safety['median_latency_ms']:.0f} ms",
    ]
    if args.llm:
        lines += ["", f"LLM picks discarded for not being in the candidate list: {llm_rejected} "
                      f"across {llm_total} requests."]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
