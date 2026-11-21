"""Genome: four integer genes (0-4) that gate how much cognition an agent may use.

Capacity genes (MEM, TALK, LORE) cost energy every lifetime. AGG is free.
Mutation is blind: it never sees what the agent experienced.
"""
import random

TRAITS = ["MEM", "TALK", "LORE", "AGG"]
CAPACITY = ["MEM", "TALK", "LORE"]
MAX_LEVEL = 4

MEM_SLOTS = [0, 2, 4, 8, 16]         # own past trial records visible
TALK_WORDS = [0, 3, 6, 12, 24]       # max words per message / gossip line
LORE_WORDS = [0, 10, 25, 50, 100]    # max words of inherited notes read or written
GRAB_PROB = [0.0, 0.2, 0.4, 0.6, 0.8]  # chance of grabbing after a joint success


def random_genome(rng: random.Random) -> dict:
    return {t: rng.randint(0, MAX_LEVEL) for t in TRAITS}


def mutate(genome: dict, rng: random.Random, p: float = 0.25) -> dict:
    child = dict(genome)
    for t in TRAITS:
        if rng.random() < p:
            child[t] = min(MAX_LEVEL, max(0, child[t] + rng.choice([-1, 1])))
    return child


def upkeep(genome: dict, cost: float) -> float:
    return cost * sum(genome[t] for t in CAPACITY)


def mem_slots(genome: dict) -> int:
    return MEM_SLOTS[genome["MEM"]]


def talk_words(genome: dict) -> int:
    return TALK_WORDS[genome["TALK"]]


def lore_words(genome: dict) -> int:
    return LORE_WORDS[genome["LORE"]]


def grab_prob(genome: dict) -> float:
    return GRAB_PROB[genome["AGG"]]


def clip_words(text, n: int) -> str:
    """Hard cap on words. Enforced in code so capacity is real, not role-play."""
    if not text or n <= 0:
        return ""
    return " ".join(str(text).split()[:n])


def visible_records(records: list, genome: dict) -> list:
    """The most recent records the agent's memory can hold."""
    n = mem_slots(genome)
    return records[-n:] if n > 0 else []
