"""CLI entry point and chat loop for Companion AI."""

import argparse
import sys

from config import (
    COLOR_BLUE,
    COLOR_DIM,
    COLOR_GREEN,
    COLOR_RESET,
    DEFAULT_MODEL,
    EVOLVE_EVERY_N_TURNS,
    MAX_CONTEXT_TURNS,
    MEMORY_BLOCK_EMPTY,
    MEMORY_BLOCK_WITH_MEMORIES,
    SYSTEM_PROMPT_TEMPLATE,
)
from extraction import extract_facts
from llm import chat_completion, check_ollama
from memory import MemoryStore
from personality import PersonalityStore


def build_system_prompt(personality, memory, user_input):
    """Assemble the system prompt from personality + retrieved memories."""
    personality_block = personality.build_personality_block()

    memories = memory.retrieve(user_input)
    if memories:
        mem_lines = "\n".join(f"- {m}" for m in memories)
        memory_block = MEMORY_BLOCK_WITH_MEMORIES.format(memories=mem_lines)
    else:
        memory_block = MEMORY_BLOCK_EMPTY

    return SYSTEM_PROMPT_TEMPLATE.format(
        personality_block=personality_block,
        memory_block=memory_block,
    )


def print_traits(personality):
    """Print all personality traits with visual bars."""
    traits = personality.get_traits()
    for name, value in traits.items():
        filled = round(value * 20)
        bar = "\u2588" * filled + "\u2591" * (20 - filled)
        print(f"  {name:12s}: {bar} {value:.2f}")


def handle_slash_command(cmd, personality, memory):
    """Handle slash commands. Return True if the command was recognized."""
    cmd = cmd.strip().lower()
    if cmd == "/traits":
        print_traits(personality)
        return True
    if cmd == "/memories":
        print(f"  Total memories stored: {memory.count()}")
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Companion AI — local conversational chatbot")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--reset", action="store_true", help="Clear all memory and personality data")
    args = parser.parse_args()

    model = args.model

    # --- Startup: verify Ollama ---
    if not check_ollama(model):
        print(
            f"\n{COLOR_BLUE}Error: Cannot connect to Ollama or model '{model}' is not available.{COLOR_RESET}\n"
            f"\nRemediation steps:\n"
            f"  1. Make sure Ollama is installed (https://ollama.com)\n"
            f"  2. Start the server:  ollama serve\n"
            f"  3. Pull the model:    ollama pull {model}\n"
            f"  4. Re-run this program.\n"
        )
        sys.exit(1)

    # --- Initialize stores ---
    memory = MemoryStore()
    personality = PersonalityStore()

    # --- Handle --reset ---
    if args.reset:
        memory.clear()
        personality.reset()
        print(f"{COLOR_DIM}All memory and personality data has been reset.{COLOR_RESET}")

    # --- Print startup info ---
    print(f"\n{COLOR_GREEN}=== Companion AI ==={COLOR_RESET}")
    print(f"{COLOR_DIM}Model: {model}{COLOR_RESET}")
    print(f"{COLOR_DIM}Personality traits:{COLOR_RESET}")
    print_traits(personality)
    print(f"{COLOR_DIM}Memories stored: {memory.count()}{COLOR_RESET}")
    print()
    print(f"Type a message to start chatting.")
    print(f"Commands: /traits, /memories")
    print(f"Type 'quit' or 'exit' to leave.\n")

    # --- Chat loop ---
    history = []
    turn_count = 0

    try:
        while True:
            try:
                user_input = input(f"{COLOR_BLUE}You: {COLOR_RESET}")
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit"):
                print("Goodbye!")
                break

            if user_input.startswith("/"):
                if handle_slash_command(user_input, personality, memory):
                    continue

            # Build system prompt
            system_prompt = build_system_prompt(personality, memory, user_input)

            # Add user message to history
            history.append({"role": "user", "content": user_input})

            # Trim history to max context window
            max_messages = MAX_CONTEXT_TURNS * 2
            trimmed = history[-max_messages:]

            # Get LLM response
            try:
                reply = chat_completion(model, system_prompt, trimmed)
            except Exception as e:
                print(f"{COLOR_DIM}Error communicating with Ollama: {e}{COLOR_RESET}")
                # Remove the user message we just appended so it doesn't corrupt history
                history.pop()
                continue

            # Add assistant reply to history
            history.append({"role": "assistant", "content": reply})
            turn_count += 1

            # Display reply
            print(f"{COLOR_GREEN}Companion:{COLOR_RESET} {reply}")

            # Memory extraction (best-effort)
            try:
                facts = extract_facts(model, user_input, reply)
                if facts:
                    added = memory.add_memories(facts)
                    if added > 0:
                        print(f"{COLOR_DIM}[+{added} new memory/memories stored]{COLOR_RESET}")
            except Exception:
                pass

            # Personality evolution (best-effort)
            try:
                if turn_count > 0 and turn_count % EVOLVE_EVERY_N_TURNS == 0:
                    recent = history[-(EVOLVE_EVERY_N_TURNS * 2):]
                    changes = personality.evolve(model, recent)
                    if changes:
                        print(f"{COLOR_DIM}[Personality evolved]{COLOR_RESET}")
                        for trait, (old, new) in changes.items():
                            direction = "\u2191" if new > old else "\u2193"
                            print(f"{COLOR_DIM}  {trait}: {old:.2f} {direction} {new:.2f}{COLOR_RESET}")
            except Exception:
                pass

    finally:
        personality.close()


if __name__ == "__main__":
    main()
