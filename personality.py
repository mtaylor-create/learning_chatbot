"""SQLite personality store with trait evolution logic."""

import json
import os
import re
import sqlite3

from config import (
    DEFAULT_PERSONALITY,
    PERSONALITY_STEP_SIZE,
    SQLITE_PATH,
    TRAIT_DEFINITIONS,
    TRAIT_MAX,
    TRAIT_MIN,
)
from llm import generate


def _clamp(value, lo=TRAIT_MIN, hi=TRAIT_MAX):
    return max(lo, min(hi, value))


class PersonalityStore:
    """Stores personality trait scores in SQLite and handles evolution."""

    def __init__(self, db_path=SQLITE_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS personality (
                   id INTEGER PRIMARY KEY CHECK (id = 1),
                   traits_json TEXT NOT NULL
               )"""
        )
        self._conn.commit()

    def get_traits(self):
        """Return the current trait dict, inserting defaults on first run."""
        row = self._conn.execute(
            "SELECT traits_json FROM personality WHERE id = 1"
        ).fetchone()
        if row is None:
            self._save_traits(DEFAULT_PERSONALITY)
            return dict(DEFAULT_PERSONALITY)
        return json.loads(row[0])

    def build_personality_block(self):
        """Convert current traits to a natural-language prompt block."""
        traits = self.get_traits()
        lines = ["Your current personality:"]
        for trait, value in traits.items():
            defn = TRAIT_DEFINITIONS.get(trait)
            if defn is None:
                continue
            if value > 0.7:
                lines.append(f"- {defn['high_prompt']}")
            elif value < 0.3:
                lines.append(f"- {defn['low_prompt']}")
            else:
                lines.append(f"- {defn['mid_prompt']}")
        return "\n".join(lines)

    def evolve(self, model, recent_messages):
        """Run personality evolution step. Return {trait: (old, new)} or None."""
        traits = self.get_traits()
        transcript = _format_transcript(recent_messages)
        prompt = _build_evolution_prompt(traits, transcript)

        raw = generate(model, prompt, temperature=0.3)
        deltas = _parse_evolution_response(raw)
        if deltas is None:
            return None

        changes = {}
        new_traits = dict(traits)
        for trait in traits:
            delta = deltas.get(trait, 0.0)
            if delta == 0.0:
                continue
            nudge = max(-PERSONALITY_STEP_SIZE, min(PERSONALITY_STEP_SIZE, delta * PERSONALITY_STEP_SIZE))
            old_val = traits[trait]
            new_val = _clamp(old_val + nudge)
            if new_val != old_val:
                new_traits[trait] = new_val
                changes[trait] = (old_val, new_val)

        if changes:
            self._save_traits(new_traits)
        return changes

    def reset(self):
        """Overwrite traits with defaults."""
        self._save_traits(DEFAULT_PERSONALITY)

    def close(self):
        """Close the SQLite connection."""
        self._conn.close()

    def _save_traits(self, traits):
        self._conn.execute(
            """INSERT INTO personality (id, traits_json) VALUES (1, ?)
               ON CONFLICT(id) DO UPDATE SET traits_json = excluded.traits_json""",
            (json.dumps(traits),),
        )
        self._conn.commit()


def _format_transcript(messages):
    """Format message dicts into a readable transcript."""
    lines = []
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Companion"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def _build_evolution_prompt(traits, transcript):
    trait_defs = "\n".join(
        f"- {name}: low = {d['low']}, high = {d['high']}"
        for name, d in TRAIT_DEFINITIONS.items()
    )
    return f"""\
You are analyzing a conversation to determine how the AI companion's personality \
should evolve based on the user's interaction style and preferences.

Current trait scores (0.0 to 1.0):
{json.dumps(traits, indent=2)}

Trait definitions:
{trait_defs}

Recent conversation:
{transcript}

Based on how the user communicates and what they seem to prefer, suggest \
adjustments to each trait. Respond with ONLY a JSON object mapping each trait \
name to a float between -1.0 and 1.0, where:
- Negative values mean the trait should decrease
- Positive values mean the trait should increase
- 0 means no change
- The magnitude indicates confidence (1.0 = very confident, 0.1 = slight nudge)

Respond with ONLY the JSON object, no other text."""


def _parse_evolution_response(raw):
    """Parse the LLM's evolution response into a delta dict, or None."""
    text = raw.strip()
    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    # Validate entries
    result = {}
    for key, val in data.items():
        if key not in DEFAULT_PERSONALITY:
            continue
        try:
            result[key] = float(val)
        except (TypeError, ValueError):
            continue
    return result if result else None
