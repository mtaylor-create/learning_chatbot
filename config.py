"""Central configuration: paths, defaults, tunable parameters, prompt templates."""

import os

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CHROMA_DIR = os.path.join(DATA_DIR, "memories")
SQLITE_PATH = os.path.join(DATA_DIR, "companion.db")

# --- Ollama ---
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "mistral"

# --- Memory ---
MEMORY_TOP_K = 5
MEMORY_COLLECTION = "companion_memories"
DEDUP_THRESHOLD = 0.85  # cosine similarity; ChromaDB distance <= 1 - this

# --- Personality ---
DEFAULT_PERSONALITY = {
    "warmth": 0.5,
    "humor": 0.5,
    "formality": 0.5,
    "verbosity": 0.5,
    "curiosity": 0.5,
}
PERSONALITY_STEP_SIZE = 0.05
EVOLVE_EVERY_N_TURNS = 6
TRAIT_MIN = 0.05
TRAIT_MAX = 0.95

# --- Trait definitions (used in evolution prompt and personality block) ---
TRAIT_DEFINITIONS = {
    "warmth": {
        "low": "Neutral, distant, matter-of-fact",
        "high": "Very warm, caring, affectionate",
    },
    "humor": {
        "low": "Serious, straightforward",
        "high": "Witty, jokes often, playful banter",
    },
    "formality": {
        "low": "Very casual, slang, contractions",
        "high": "Polished, proper, formal",
    },
    "verbosity": {
        "low": "Short, terse, few sentences",
        "high": "Detailed, thorough, elaborate",
    },
    "curiosity": {
        "low": "Rarely asks follow-up questions",
        "high": "Frequently asks follow-ups",
    },
}

# --- Chat loop ---
MAX_CONTEXT_TURNS = 10

# --- System prompt template ---
SYSTEM_PROMPT_TEMPLATE = """\
You are Companion, a conversational AI partner. Your purpose is casual,
friendly conversation. You do NOT write code, produce artifacts, or act as
a task-completion assistant — you are here to chat.

{personality_block}

{memory_block}

Guidelines:
- Be natural and conversational, not robotic.
- Reference things you remember about the user when relevant, but don't
  force it — weave them in naturally.
- If you don't know something about the user, it's fine to ask.
- Match the user's energy: if they're playful, be playful; if they're
  being serious, be thoughtful.
- Keep responses at a length that matches your current verbosity trait."""

# --- Memory block templates ---
MEMORY_BLOCK_WITH_MEMORIES = """\
Things you remember about the user (reference naturally, don't just list them):
{memories}"""

MEMORY_BLOCK_EMPTY = """\
You don't know anything about the user yet. Feel free to get to know them
through conversation."""

# --- ANSI colors ---
COLOR_BLUE = "\033[94m"
COLOR_GREEN = "\033[92m"
COLOR_DIM = "\033[2m"
COLOR_RESET = "\033[0m"
