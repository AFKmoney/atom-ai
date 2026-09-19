"""Chat speech helpers for ATOM. Not a language model. Gate only."""

from __future__ import annotations


def format_dialogue_prompt(prompt: str) -> str:
    raw = (prompt or "").strip()
    if not raw:
        return "Utilisateur:\nAssistant: "
    if "Utilisateur:" in raw or "Assistant:" in raw:
        return raw if raw.endswith("Assistant:") or raw.endswith("Assistant: ") else raw
    return f"Utilisateur: {raw}\nAssistant: "


_SPEECH_HINTS = frozenset(
    {
        "je", "tu", "il", "elle", "on", "nous", "vous", "est", "suis", "es",
        "les", "des", "une", "un", "le", "la", "et", "que", "qui", "pas",
        "bonjour", "salut", "oui", "non", "merci", "comment", "vais", "va",
        "fait", "plus", "dans", "pour", "avec", "hello", "ok", "ca", "moi", "toi",
    }
)


def speech_ok(payload: bytes) -> bool:
    if not payload:
        return False
    text = payload.decode("utf-8", errors="replace").strip()
    if len(text) < 2:
        return False
    letters = [ch for ch in text.lower() if ch.isalpha()]
    if len(letters) < 2:
        return False
    vowels = set("aeiouyàâäéèêëïîôùûüç")
    if not any(ch in vowels for ch in letters):
        return False
    printable = sum(ch.isprintable() or ch in "\n\t" for ch in text) / max(len(text), 1)
    if printable < 0.8:
        return False
    if len(set(letters)) == 1:
        return False
    tokens = []
    buf: list[str] = []
    for ch in text.lower():
        if ch.isalpha():
            buf.append(ch)
        elif buf:
            tokens.append("".join(buf))
            buf = []
    if buf:
        tokens.append("".join(buf))
    if any(tok in _SPEECH_HINTS for tok in tokens):
        return True
    return 2 <= len(text) <= 6
