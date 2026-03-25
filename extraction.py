"""Fact extraction from conversation turns via LLM."""

import re

from llm import generate

EXTRACTION_PROMPT = """\
You are a memory extraction system. Given a conversation exchange between a user \
and an AI companion, extract any NEW facts, preferences, or personal details \
the user revealed.

Rules:
- Only extract facts stated or strongly implied by the USER (not the AI's own statements).
- Write each fact as a short, standalone, third-person sentence \
(e.g., "The user's name is Jordan.").
- Phrase corrected facts as the corrected version.
- Respond with exactly NONE if there are no new facts.
- Output one fact per line, no numbering or bullet points.

User said: {user_msg}

Companion replied: {assistant_msg}

Extracted facts:"""


def extract_facts(model, user_msg, assistant_msg):
    """Ask the LLM to extract facts from a conversation turn.

    Returns:
        A list of fact strings, possibly empty.
    """
    prompt = EXTRACTION_PROMPT.format(user_msg=user_msg, assistant_msg=assistant_msg)
    raw = generate(model, prompt, temperature=0.3)
    return _parse_facts(raw)


def _parse_facts(raw):
    """Parse the LLM's fact extraction response into a list of strings."""
    text = raw.strip()
    if not text or text.upper().startswith("NONE"):
        return []

    facts = []
    for line in text.splitlines():
        line = line.strip()
        # Strip leading bullets, dashes, or numbering
        line = re.sub(r"^[\-\*•]\s*", "", line)
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        line = line.strip()
        if len(line) < 6:
            continue
        facts.append(line)
    return facts
