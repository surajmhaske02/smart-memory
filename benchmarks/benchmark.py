"""Naive "stuff everything into the prompt" vs smart_memory Context Cards.

Synthetic but reproducible: several users, each with hundreds of facts, events and
procedures, plus questions whose answer is one specific memory. We measure prompt
tokens, whether the answer made it into the prompt (recall), cross-user leakage and
latency.

    python benchmarks/benchmark.py            # default sizes
    python benchmarks/benchmark.py --users 5 --per-user 1000
"""

from __future__ import annotations

import argparse
import random
import re
import statistics
import time

from smart_memory import Memory
from smart_memory.text import estimate_tokens

CITIES = ["Pune", "Berlin", "Austin", "Lagos", "Osaka", "Lima", "Oslo", "Perth", "Quito", "Cairo"]
FOODS = ["Thai curry", "sushi", "tacos", "falafel", "ramen", "paneer tikka", "pho", "paella"]
PRODUCTS = ["router X200", "laptop Pro 14", "phone Z5", "monitor U27", "keyboard K3", "camera R8"]
TOOLS = ["pricing API", "refund API", "shipping API", "CRM sync", "invoice service", "search index"]
TOPICS = ["billing", "shipping delays", "account security", "warranty", "discount codes", "data export"]


def build_user(rng: random.Random, name: str, n: int):
    """Return (memories, questions). Each question is (query, substring that must appear in the prompt)."""
    city, food = rng.choice(CITIES), rng.choice(FOODS)
    product, tool = rng.choice(PRODUCTS), rng.choice(TOOLS)
    order = rng.randint(10_000, 99_999)
    gold = [
        (f"{name} lives in {city}", f"Which city does {name} live in?", city),
        (f"{name}'s favourite food is {food}", f"What food does {name} like?", food),
        (f"{name} bought the {product} last month", f"Which product did {name} buy?", product),
        (f"The {tool} call for {name}'s order {order} failed with a timeout",
         f"Did anything fail on order {order}?", str(order)),
        (f"{name} prefers to be contacted by email, never by phone", f"How should we contact {name}?", "email"),
    ]
    memories = [g[0] for g in gold]
    while len(memories) < n:
        kind = rng.random()
        if kind < 0.45:
            memories.append(f"{name} asked about {rng.choice(TOPICS)} on ticket {rng.randint(1000, 9999)}")
        elif kind < 0.75:
            memories.append(f"Support agent resolved a {rng.choice(TOPICS)} question for {name} "
                            f"after {rng.randint(2, 40)} minutes")
        elif kind < 0.9:
            memories.append(f"{name} viewed the {rng.choice(PRODUCTS)} product page {rng.randint(2, 9)} times")
        else:
            memories.append(f"Note {rng.randint(1, 10**6)}: {name} mentioned the weather in {rng.choice(CITIES)}")
    rng.shuffle(memories)
    return memories, [(q, needle) for _, q, needle in gold]


def run(users: int, per_user: int, budget: int, seed: int) -> None:
    rng = random.Random(seed)
    names = [f"User{i}" for i in range(users)]
    mem = Memory(trace=False)
    data = {}
    t0 = time.perf_counter()
    for name in names:
        memories, questions = build_user(rng, name, per_user)
        scoped = mem.scope(user_id=name, agent_id="support")
        for text in memories:
            scoped.add(text)
        data[name] = (memories, questions)
    ingest_s = time.perf_counter() - t0

    naive_tokens, card_tokens, latencies = [], [], []
    hits = leaks = total = 0
    for name in names:
        memories, questions = data[name]
        scoped = mem.scope(user_id=name, agent_id="support")
        naive_prompt = "\n".join(f"- {m}" for m in memories)
        for query, needle in questions:
            start = time.perf_counter()
            card = scoped.context(query, max_tokens=budget)
            latencies.append((time.perf_counter() - start) * 1000)
            text = card.render()
            naive_tokens.append(estimate_tokens(naive_prompt))
            card_tokens.append(card.token_estimate)
            hits += needle in text
            leaks += any(re.search(rf"\b{other}\b", text) for other in names if other != name)
            total += 1

    naive, smart = statistics.mean(naive_tokens), statistics.mean(card_tokens)
    print(f"users={users} memories/user={per_user} budget={budget} queries={total}")
    print(f"ingest: {users * per_user / ingest_s:,.0f} memories/s")
    print(f"naive prompt tokens (mean):  {naive:,.0f}   recall 100.0% (everything is in the prompt)")
    print(f"context card tokens (mean):  {smart:,.0f}   recall {100 * hits / total:.1f}%")
    print(f"token reduction:             {100 * (1 - smart / naive):.1f}%")
    print(f"cross-user leaks:            {leaks}")
    print(f"context latency p50/p95:     {statistics.median(latencies):.1f} / "
          f"{sorted(latencies)[int(0.95 * (len(latencies) - 1))]:.1f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--users", type=int, default=3)
    parser.add_argument("--per-user", type=int, default=300)
    parser.add_argument("--budget", type=int, default=300, help="context card token budget")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    run(args.users, args.per_user, args.budget, args.seed)
