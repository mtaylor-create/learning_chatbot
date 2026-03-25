"""Ollama API wrapper for chat completions and single-shot generation."""

import requests
from config import OLLAMA_BASE_URL


def chat_completion(model, system_prompt, messages, temperature=0.7):
    """Send a chat request to Ollama and return the assistant's reply string.

    Args:
        model: Ollama model name.
        system_prompt: System prompt injected as the first message.
        messages: List of {"role": "user"|"assistant", "content": "..."} dicts.
        temperature: Sampling temperature.

    Returns:
        The assistant's reply text.
    """
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": model,
            "messages": full_messages,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def generate(model, prompt, temperature=0.3):
    """Send a single-shot generation request to Ollama.

    Used for structured tasks (memory extraction, personality evaluation)
    where lower temperature produces more consistent output.

    Returns:
        The response text.
    """
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def check_ollama(model):
    """Verify Ollama is reachable and the requested model is available.

    Returns:
        True only if the server responds and the model is listed.
    """
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        resp.raise_for_status()
        available = [m["name"] for m in resp.json().get("models", [])]
        # Ollama may list models as "mistral:latest" — match with or without tag
        return any(
            name == model or name.startswith(f"{model}:")
            for name in available
        )
    except Exception:
        return False
