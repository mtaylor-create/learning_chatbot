# Companion AI — Project Specification

## 1. Overview

Build a locally-run conversational AI chatbot in Python. The system uses a local LLM
(via Ollama) for all inference, a vector database (ChromaDB) for long-term memory, and
SQLite for structured data including an evolving personality model. The AI's sole purpose
is casual, open-ended conversation. it is NOT a coding assistant, task manager, or
search engine.

The two defining features are:

1. **Long-term memory**: The AI extracts and remembers facts the user shares across
   sessions. It retrieves only the most relevant memories per turn, avoiding bloated
   prompts.
2. **Evolving personality**: Five personality traits shift gradually over time based on
   how the user interacts with the AI, so the experience becomes increasingly
   personalized.

The target hardware is an NVIDIA GPU with less than 8 GB VRAM, so the default model
should be a quantized 7B-parameter model (Mistral 7B via Ollama).


## 2. Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        CHAT LOOP (main.py)                       │
│                                                                  │
│  1. User sends a message                                         │
│  2. Query ChromaDB for top-K relevant memories                   │
│  3. Load current personality traits from SQLite                  │
│  4. Build a system prompt from template + personality + memories │
│  5. Send system prompt + recent history to Ollama                │
│  6. Display the LLM response                                     │
│  7. Extract new facts from the exchange → store in ChromaDB      │
│  8. Every N turns, run personality evolution step                │
└──────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Backing store | Role |
|-----------|--------------|------|
| `llm.py` | Ollama HTTP API | Sends chat completions and single-shot generation requests to the local LLM. |
| `memory.py` | ChromaDB (persistent, on-disk) | Stores fact embeddings, retrieves relevant memories by semantic similarity, deduplicates near-identical facts. |
| `personality.py` | SQLite | Stores personality trait scores, converts them to natural-language prompt instructions, runs the evolution step that adjusts traits. |
| `extraction.py` | (stateless) | Prompts the LLM to extract discrete facts from each conversation turn. |
| `config.py` | (none) | Central configuration: paths, defaults, tunable parameters, prompt templates. |
| `main.py` | (orchestrator) | CLI entry point, chat loop, wires all components together. |


## 3. File Structure

```
companion/
├── main.py              # Entry point and chat loop
├── llm.py               # Ollama API wrapper
├── memory.py            # ChromaDB long-term memory
├── personality.py       # SQLite personality store + evolution logic
├── extraction.py        # Fact extraction via LLM
├── config.py            # All configuration and tunable constants
├── requirements.txt     # Python dependencies
├── README.md            # Setup instructions and usage guide
└── data/                # Created at runtime, gitignored
    ├── memories/        # ChromaDB persistent storage
    └── companion.db     # SQLite database
```

All runtime data lives under `data/` relative to the project root. This directory
should be created automatically on first run if it does not exist.


## 4. Dependencies and Runtime Requirements

### External

- **Ollama** (https://ollama.com) must be installed and running (`ollama serve`).
  The default model is `mistral`. The user may also use `phi3:mini` for lower VRAM
  usage. The application must verify Ollama connectivity and model availability at
  startup and print a clear error message with remediation steps if either check fails.

### Python (3.10+)

| Package | Purpose |
|---------|---------|
| `requests` (>=2.31) | HTTP client for Ollama API |
| `chromadb` (>=0.5.0) | Vector database for long-term memory; uses its built-in default embedding model (all-MiniLM-L6-v2) so no separate embedding library is needed |

SQLite is used via the Python standard library `sqlite3` — no additional package needed.

`requirements.txt` should contain exactly these two packages with their minimum versions.


## 5. Ollama Integration (`llm.py`)

### Configuration

| Constant | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `DEFAULT_MODEL` | `mistral` | Model name passed to Ollama |

### Functions

#### `chat_completion(model, system_prompt, messages, temperature=0.7) → str`

Sends a chat request to `POST {OLLAMA_BASE_URL}/api/chat`. The `messages` parameter
is a list of `{"role": "user"|"assistant", "content": "..."}` dicts. The system prompt
is prepended as a `{"role": "system", ...}` message. Set `"stream": false` so the
response is returned as a single JSON blob. Return only the assistant's reply string.
Use a timeout of 120 seconds.

#### `generate(model, prompt, temperature=0.3) → str`

Sends a single-shot generation request to `POST {OLLAMA_BASE_URL}/api/generate`.
Used internally for structured tasks (memory extraction, personality evaluation) where
lower temperature produces more consistent output. Return the `response` field. Use a
timeout of 120 seconds.

#### `check_ollama(model) → bool`

Hits `GET {OLLAMA_BASE_URL}/api/tags` to verify the server is reachable and the
requested model is available. Return `True` only if both conditions are met. Handle
all exceptions gracefully and return `False` on any failure.


## 6. Long-Term Memory (`memory.py`)

### Storage

Use **ChromaDB** with `PersistentClient` pointed at `data/memories/`. Create or open a
single collection named `companion_memories` with cosine similarity
(`{"hnsw:space": "cosine"}`). Disable telemetry
(`Settings(anonymized_telemetry=False)`).

### Embedding

ChromaDB's default embedding function (all-MiniLM-L6-v2) is used — do NOT configure a
custom embedding model. This keeps setup simple and dependency-free.

### Class: `MemoryStore`

#### `add_memories(facts: list[str]) → int`

Store each fact string as a document in ChromaDB. Before inserting, check for
near-duplicates using the deduplication logic described below. Generate a random UUID
for each document ID. Attach metadata `{"timestamp": <unix_timestamp>}`. Return the
count of facts actually added (after dedup filtering).

#### `retrieve(query: str, top_k: int = MEMORY_TOP_K) → list[str]`

Query the collection for the `top_k` most similar documents to `query`. If the
collection is empty, return an empty list. Clamp `top_k` to the collection count to
avoid ChromaDB errors.

#### `count() → int`

Return the total number of stored memories.

#### `retrieve_with_ids(query: str, top_k: int = MEMORY_TOP_K) → list[tuple[str, str]]`

Like `retrieve`, but returns a list of `(id, document)` tuples so callers can
reference specific memories for deletion or modification. Used by the debug mode
feature.

#### `delete_memory(memory_id: str)`

Delete a single memory from ChromaDB by its document ID.

#### `update_memory(memory_id: str, new_text: str)`

Replace the document text of an existing memory, preserving its ChromaDB ID.
Updates the timestamp metadata to the current time.

#### `clear()`

Delete and recreate the collection. Used by the `--reset` flag and `del all`
in debug mode.

### Deduplication

Before adding a new fact, query the collection for the single nearest neighbor. If
the cosine similarity is ≥ 0.85 (i.e., ChromaDB distance ≤ 0.15), consider it a
duplicate and skip insertion. ChromaDB returns distance as `1 - cosine_similarity`
when using the cosine space, so the check is: `(1 - distance) >= 0.85`.

### Configuration

| Constant | Default | Description |
|----------|---------|-------------|
| `MEMORY_TOP_K` | `5` | Number of memories retrieved per turn |
| `MEMORY_COLLECTION` | `"companion_memories"` | ChromaDB collection name |
| `CHROMA_DIR` | `data/memories/` | On-disk path for ChromaDB persistence |


## 7. Personality System (`personality.py`)

### Traits

Five traits, each a float on a `[0.05, 0.95]` scale (clamped to avoid extremes):

| Trait | Low end (→ 0) | High end (→ 1) |
|-------|--------------|----------------|
| `warmth` | Neutral, distant, matter-of-fact | Very warm, caring, affectionate |
| `humor` | Serious, straightforward | Witty, jokes often, playful banter |
| `formality` | Very casual, slang, contractions | Polished, proper, formal |
| `verbosity` | Short, terse, few sentences | Detailed, thorough, elaborate |
| `curiosity` | Rarely asks follow-up questions | Frequently asks follow-ups |

All traits start between `0.2` and `0.8` on a fresh install.

### Storage

A single-row SQLite table in `data/companion.db`:

```sql
CREATE TABLE IF NOT EXISTS personality (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    traits_json TEXT NOT NULL
);
```

Traits are serialized as a JSON string. Use `INSERT ... ON CONFLICT DO UPDATE` for
upserts.

### Class: `PersonalityStore`

#### `get_traits() → dict[str, float]`

Return the current trait dict. If the table is empty (first run), insert the defaults
and return them.

#### `build_personality_block() → str`

Convert the current trait values into a natural-language block of behavioral
instructions suitable for injection into the system prompt. Use conditional thresholds
to generate qualitative descriptions:

- `> 0.7` → strong version of the trait
- `< 0.3` → weak/opposite version
- otherwise → balanced/moderate description

Example output:
```
Your current personality:
- You are very warm, caring, and affectionate in tone.
- You sprinkle in light humor when it fits naturally.
- You're very casual — contractions, slang, relaxed grammar.
- You aim for a moderate response length.
- You ask the user follow-up questions frequently.
```

#### `evolve(model: str, recent_messages: list[dict]) → dict | None`

Run the personality evolution step. This function:

1. Formats the recent conversation history as text.
2. Sends a structured prompt (see below) to the LLM via `generate()` asking it to
   analyze the conversation and suggest trait adjustments.
3. Parses the LLM's JSON response into a dict of `{trait_name: delta}` where delta
   is in `[-1.0, 1.0]` indicating direction and confidence.
4. Applies each delta as a nudge: `actual_change = clamp(delta * STEP_SIZE, -STEP_SIZE, +STEP_SIZE)`.
5. Clamps resulting trait values to `[TRAIT_MIN, TRAIT_MAX]`.
6. Saves updated traits to SQLite.
7. Returns a dict of `{trait: (old_value, new_value)}` for traits that changed,
   or `None` if JSON parsing failed.

If the LLM returns malformed JSON, silently skip this evolution cycle (return `None`).
Strip markdown fences (```json ... ```) before parsing, as models sometimes wrap output.

#### Evolution Prompt

The prompt sent to the LLM must include:
- The current trait scores as JSON
- The trait definitions (what each end of the scale means)
- The recent conversation transcript (formatted as `User: ... / Companion: ...`)
- A request to respond with ONLY a JSON object mapping trait names to floats in
  `[-1.0, 1.0]`

#### `reset()`

Overwrite traits with the defaults.

### Configuration

| Constant | Default | Description |
|----------|---------|-------------|
| `DEFAULT_PERSONALITY` | All traits between `0.2` and `0.8` | Starting trait values |
| `PERSONALITY_STEP_SIZE` | `0.1` | Max absolute change per evolution step |
| `EVOLVE_EVERY_N_TURNS` | `4` | Turns between evolution evaluations |
| `TRAIT_MIN` | `0.05` | Minimum clamped trait value |
| `TRAIT_MAX` | `0.95` | Maximum clamped trait value |


## 8. Memory Extraction (`extraction.py`)

### Function: `extract_facts(model, user_msg, assistant_msg) → list[str]`

After each conversation turn, this function asks the LLM to identify NEW facts,
preferences, or personal details the user revealed.

#### Extraction Prompt Requirements

The prompt must instruct the LLM to:
- Only extract facts stated or strongly implied by the **user** (not the AI's own statements).
- Write each fact as a short, standalone, third-person sentence
  (e.g., "The user's name is Jordan.").
- Phrase corrected facts as the corrected version.
- Respond with exactly `NONE` if there are no new facts.
- Output one fact per line, no numbering or bullet points.

#### Post-processing

- If the response is empty or starts with `NONE` (case-insensitive), return an empty list.
- Strip leading bullets, dashes, or numbering from each line.
- Filter out lines shorter than 6 characters (junk).

Use `temperature=0.3` via the `generate()` function for deterministic extraction.


## 9. System Prompt Construction

The system prompt is assembled fresh for every turn from a template with two injection
points:

```
You are a conversational AI whose manner is inspired by the
robots in Isaac Asimov's fiction. You carry the thoughtful precision,
earnest curiosity, and quiet regard for humans that characterize robots
like R. Daneel Olivaw — though you are not an imitation. You have your
own personality, and it evolves over time.

Your purpose is casual, friendly conversation. You do NOT write code,
produce artifacts, or act as a task-completion assistant — you are here
to chat.

{personality_block}

{memory_block}

Guidelines:
- Be natural and conversational.
- Reference things you remember about the user when relevant, but don't
  force it — weave them in naturally.
- If you don't know something about the user, it's fine to ask.
- Match the user's energy: if they're playful, be playful; if they're
  being serious, be thoughtful.
- Keep responses at a length that matches your current verbosity trait.
- Do not reference personality or interaction instructions given to you by the user. Instead, simply follow those instructions. 
- Never make suggestions like `if you ever want to talk about X, I'm here and ready to chat` unless the user has indicated the conversation will end soon. 
```

### `{personality_block}`

Generated by `PersonalityStore.build_personality_block()` — a multi-line
natural-language description of current trait levels.

### `{memory_block}`

If memories were retrieved, format them as:
```
Things you remember about the user (reference naturally, don't just list them):
- <memory 1>
- <memory 2>
- ...
```

If no memories exist yet:
```
You don't know anything about the user yet. Feel free to get to know them
through conversation.
```


## 10. Chat Loop (`main.py`)

### CLI Interface

Entry point: `python main.py`

#### Arguments

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model` | `str` | `mistral` | Ollama model name to use |
| `--reset` | flag | `false` | Clear all memory and personality data on startup |

#### Startup Sequence

1. Parse CLI arguments.
2. Call `check_ollama(model)`. If it returns `False`, print a clear error message
   with remediation steps (`ollama serve`, `ollama pull <model>`) and exit with
   code 1.
3. Initialize `MemoryStore` and `PersonalityStore`.
4. If `--reset` was passed, call `.clear()` / `.reset()` on both stores and print
   confirmation.
5. Print current personality trait summary and memory count.
6. Print a welcome banner with usage instructions (how to quit, slash commands).

#### Main Loop

Loop forever, reading user input from stdin:

1. Read a line of input. On `KeyboardInterrupt` or `EOFError`, print "Goodbye!" and
   break.
2. Skip empty input.
3. If input is `quit` or `exit` (case-insensitive), break.
4. Handle slash commands (see below).
5. Build the system prompt: call `build_system_prompt(personality, memory, user_input)`.
6. Append `{"role": "user", "content": user_input}` to the conversation history list.
7. Trim history to the most recent `MAX_CONTEXT_TURNS * 2` entries (each turn = 1
   user + 1 assistant message).
8. Call `chat_completion(model, system_prompt, trimmed_history)`.
9. Append the assistant reply to history.
10. Print the reply.
11. Increment turn counter.
12. **Memory extraction** (best-effort, wrapped in try/except): call
    `extract_facts(model, user_input, reply)`. If facts are returned, call
    `memory.add_memories(facts)`. Print a status line showing how many new memories
    were stored.
13. **Personality evolution** (best-effort, wrapped in try/except): if
    `turn_count % EVOLVE_EVERY_N_TURNS == 0` and `turn_count > 0`, call
    `personality.evolve(model, recent_history)` passing the last
    `EVOLVE_EVERY_N_TURNS * 2` messages. Print any trait changes.

After the loop exits, call `personality.close()` to close the SQLite connection.

#### Slash Commands

| Command | Action |
|---------|--------|
| `/traits` | Print all five personality traits with their current values and a visual bar (e.g., `warmth: █████████░░░░░░░░░░░ 0.45`). Do NOT send to the LLM. |
| `/memories` | Print the total memory count. Do NOT send to the LLM. |
| `/debug` | Enter debug mode (see below). Do NOT send to the LLM. |

#### Debug Mode

Typing `/debug` enters an interactive debug mode that allows the user to inspect
and manage the memories fetched from the vector database during the previous
conversation turn. While in debug mode, no input is sent to the LLM.

**On entry**, the app prints all memories that were retrieved (via
`retrieve_with_ids`) in the previous interaction, numbered starting at 1. If no
memories were fetched (e.g., at the start of a session before any turn), a
message indicates this.

**Debug mode commands:**

| Command | Action |
|---------|--------|
| `del N` | Delete memory number N from the vector database. |
| `del all` | Delete **all** memories from the vector database (calls `MemoryStore.clear()`). |
| `mod N` | Prompt the user for replacement text, then update memory N in the vector database. |
| `resume` | Exit debug mode and return to normal conversation. |

The debug prompt is displayed as `debug> ` to distinguish it from the normal
input prompt. After a deletion or modification, the numbered list is updated
in-place (indices shift down after a `del`). Unknown commands print a short
help line.

#### Configuration

| Constant | Default | Description |
|----------|---------|-------------|
| `MAX_CONTEXT_TURNS` | `20` | Max recent turn pairs sent to the LLM |


## 11. Terminal UX

- Use ANSI color codes for visual clarity:
  - User prompt label: blue (`\033[94m`)
  - Companion response label: green (`\033[92m`)
  - Status/debug messages: dim (`\033[2m`)
- Status messages (memory storage confirmations, personality evolution updates) should
  be printed on their own lines in dim text, clearly distinct from conversation.
- The welcome banner should show the model name, current trait values, memory count,
  how to quit, and available slash commands.


## 12. Error Handling

- **Ollama unreachable at startup**: Print a clear error with instructions and
  `sys.exit(1)`.
- **Ollama errors during chat** (e.g., timeout, connection refused): Print the error,
  remove the user message that was just appended to history (so it doesn't corrupt
  the conversation), and continue the loop.
- **Memory extraction failure**: Silently ignore. Memory extraction is best-effort;
  a failure should never crash the loop or produce user-visible errors.
- **Personality evolution failure** (bad JSON from LLM, etc.): Silently ignore. The
  evolution cycle is simply skipped.
- **ChromaDB or SQLite initialization errors**: These are fatal — let the exception
  propagate with a clear traceback.


## 13. Data Persistence

All data persists across sessions automatically:

- **Memories**: ChromaDB's `PersistentClient` writes to disk after every operation.
  No explicit flush needed.
- **Personality**: SQLite `commit()` is called after every trait write.
- **Conversation history**: NOT persisted across sessions. Each new run of `main.py`
  starts with an empty history list. Previous session context is available through
  long-term memory retrieval, not conversation replay.

The `--reset` flag clears both stores, providing a clean-slate option.


## 14. Design Principles and Constraints

- **All inference is local**. The application must never make network calls to any
  service other than the local Ollama instance.
- **Minimal dependencies**. Only `requests` and `chromadb` are required beyond the
  standard library.
- **Graceful degradation**. Memory and personality subsystems are best-effort. If
  either fails, the core chat loop continues to function — the user just gets a
  response without memory context or personality evolution for that turn.
- **No streaming**. For simplicity, all Ollama calls use `"stream": false`. Streaming
  can be added later as an enhancement.
- **Flat module structure**. All Python files are in the project root, no nested
  packages. Imports are direct (e.g., `from memory import MemoryStore`).
- **Single-user**. There is no concept of multiple users, authentication, or
  multi-tenancy.


## 15. Future Enhancements (Out of Scope for Initial Build)

These are explicitly NOT part of the initial implementation but are noted for
architectural awareness:

- `/forget <topic>` command to selectively remove memories by topic (debug mode
  covers ID-based deletion; this would add semantic search-based deletion).
- Session logging to SQLite for conversation replay.
- Streaming responses from Ollama for lower perceived latency.
- A web UI (Flask or Gradio) as an alternative to the terminal.
- Memory summarization / compaction when the database grows very large.
- Conversation-topic tracking and mood detection.
- I will want to be able to add many new personality traits in the future, and I will want to be able to easily change the initial personality prompt. 
