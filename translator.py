"""Fordítás modul — angol szegmensek magyarra fordítása OpenAI GPT vagy Ollama LLM-mel."""

import os
import logging
import time
import json
from typing import List
from dataclasses import dataclass

logger = logging.getLogger("hu_dub")

TRANSLATION_SYSTEM_PROMPT = """Te egy professzionális szinkronfordító vagy, aki angol technikai videókat fordít magyarra.

Szabályok:
- Természetes, beszélt magyar nyelvet használj (nem irodalmi, hanem szinkronhanghoz illő stílus)
- Technikai kifejezéseket tartsd meg angolul: MCP, .NET, Azure, C#, API, SDK, GitHub, Docker, Kubernetes, JSON, REST, HTTP, URL, stb.
- Tulajdonneveket NE fordítsd: ChatGPT, Claude, Cursor, Airtable, Microsoft, stb.
- Rövidítéseket tartsd meg: AI, LLM, UI, stb.
- A fordítás hossza legyen hasonló az eredetihez (szinkronhoz kell illeszkednie)
- Tegező formát használj

Formátum: Minden sor "SORSZÁM|fordított szöveg" formátumban. A sorszámokat pontosan tartsd meg."""

RECOMMENDED_OLLAMA_MODELS = [
    "gemma3:27b",
    "qwen3:30b-a3b",
    "deepseek-v3:32b",
    "towerinstruct:13b",
    "gemma3:12b",
    "mistral:7b",
]


@dataclass
class TranslatedSegment:
    start: float
    end: float
    text_en: str
    text_hu: str


def translate_segments(
    segments: list,
    translator: str = "openai",
    api_key: str = "",
    model: str = "gpt-4o",
    ollama_url: str = "http://localhost:11434",
    max_retries: int = 3,
) -> List[TranslatedSegment]:
    if translator == "ollama":
        return _translate_with_ollama(segments, model, ollama_url, max_retries)
    else:
        return _translate_with_openai(segments, api_key, model, max_retries)


def _translate_with_openai(segments, api_key, model, max_retries):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    batch_size = 20
    all_translated = []
    for batch_start in range(0, len(segments), batch_size):
        batch = segments[batch_start:batch_start + batch_size]
        translated_batch = _call_llm_batch(
            batch, model, max_retries,
            call_fn=lambda msgs, m: _openai_chat(client, msgs, m),
        )
        all_translated.extend(translated_batch)
    logger.info(f"Fordítás kész (OpenAI {model}): {len(all_translated)} szegmens")
    return all_translated


def _translate_with_ollama(segments, model, ollama_url, max_retries):
    import urllib.request
    logger.info(f"Ollama fordítás: {model} @ {ollama_url}")
    try:
        req = urllib.request.Request(f"{ollama_url}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            available = [m["name"] for m in data.get("models", [])]
            model_base = model.split(":")[0]
            found = any(model_base in m for m in available)
            if not found:
                logger.warning(f"'{model}' nem található az Ollama-ban. Elérhető: {', '.join(available[:10])}")
                logger.warning(f"Töltsd le: ollama pull {model}")
    except Exception as e:
        raise RuntimeError(f"Ollama nem elérhető: {ollama_url} — {e}\nIndítsd el: ollama serve")

    def ollama_chat(messages, mdl):
        return _ollama_chat_request(ollama_url, messages, mdl)

    batch_size = 15
    all_translated = []
    for batch_start in range(0, len(segments), batch_size):
        batch = segments[batch_start:batch_start + batch_size]
        translated_batch = _call_llm_batch(batch, model, max_retries, call_fn=ollama_chat)
        all_translated.extend(translated_batch)
    logger.info(f"Fordítás kész (Ollama {model}): {len(all_translated)} szegmens")
    return all_translated


def _ollama_chat_request(base_url, messages, model):
    import urllib.request
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.3},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
        return data["message"]["content"]


def _openai_chat(client, messages, model):
    response = client.chat.completions.create(model=model, messages=messages, temperature=0.3)
    return response.choices[0].message.content.strip()


def _call_llm_batch(segments, model, max_retries, call_fn):
    numbered_lines = [f"{i}|{seg.text}" for i, seg in enumerate(segments)]
    input_text = "\n".join(numbered_lines)
    messages = [
        {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
        {"role": "user", "content": f"Fordítsd le az alábbi angol szövegszegmenseket magyarra:\n\n{input_text}"},
    ]
    for attempt in range(max_retries):
        try:
            response_text = call_fn(messages, model)
            return _parse_translation_response(response_text, segments)
        except Exception as e:
            logger.warning(f"Fordítás hiba (próba {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"Fordítás sikertelen {max_retries} próba után: {e}")
    return []


def _parse_translation_response(response_text, segments):
    translations = {}
    for line in response_text.strip().split("\n"):
        line = line.strip()
        if "|" in line:
            parts = line.split("|", 1)
            try:
                idx = int(parts[0].strip())
                text = parts[1].strip()
                translations[idx] = text
            except (ValueError, IndexError):
                continue
    result = []
    for i, seg in enumerate(segments):
        hu_text = translations.get(i, seg.text)
        if i not in translations:
            logger.warning(f"Hiányzó fordítás a(z) {i}. szegmenshez, eredeti szöveg használata")
        result.append(TranslatedSegment(
            start=seg.start, end=seg.end, text_en=seg.text, text_hu=hu_text,
        ))
        logger.debug(f"  [{seg.start:.1f}-{seg.end:.1f}] {seg.text} → {hu_text}")
    return result
