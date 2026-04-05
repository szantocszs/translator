"""Fordítás modul — angol szegmensek magyarra fordítása OpenAI GPT vagy Ollama LLM-mel."""

import os
import logging
import time
import json
import re
from typing import List
from dataclasses import dataclass, field

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


@dataclass
class NaturalizedGroup:
    """Egy természetesített szegmenscsoport — több eredeti szegmens összevonva."""
    start: float
    end: float
    text_en: str          # eredeti angol szöveg (összefűzve)
    text_hu_original: str  # első-körös fordítás (összefűzve)
    text_hu_natural: str   # LLM által átírt természetes magyar szöveg
    source_indices: List[int] = field(default_factory=list)


NATURALIZE_SYSTEM_PROMPT = """Te egy professzionális magyar szinkronszerkesztő vagy. A feladatod, hogy egy első-körös magyar fordítást természetesebb, folyékonyabb magyar szöveggé alakíts.

Szabályok:
- A szövegnek természetesen, gördülékenyen kell hangoznia, mintha egy anyanyelvi magyar beszélné
- Nyelvtanilag, stilisztikailag és olvasásra sokkal természetesebben hangozzon
- Nem kell pontosan egy az egyben ugyanaz legyen — a lényeg a természetesség
- Tegező formát használj
- Technikai kifejezéseket tartsd meg angolul: MCP, .NET, Azure, C#, API, SDK, GitHub, Docker, Kubernetes, JSON, REST, HTTP, URL, stb.
- Tulajdonneveket NE fordítsd: ChatGPT, Claude, Cursor, Airtable, Microsoft, stb.
- Rövidítéseket tartsd meg: AI, LLM, UI, stb.
- NE adj hozzá és NE hagyj ki tényeket, információkat — csak a megfogalmazás változzon
- A szöveg terjedelme maradjon hasonló az eredetihez (beszélt szöveg, időhöz kell illeszkednie)
- A célhossz körülbelül {target_seconds:.0f} másodpercnyi beszédnek felel meg

Az eredeti angol szöveg referenciaként szolgál — használd a jelentés pontos megőrzéséhez.

Válaszolj CSAK a természetesített magyar szöveggel, semmi mással."""


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


# ─── Naturalizálás (természetesebb magyar szinkron) ─────────────────────────

def _group_segments_by_gaps(
    segments: List[TranslatedSegment],
    gap_threshold: float = 1.5,
    max_group_duration: float = 60.0,
    max_group_chars: int = 500,
) -> List[List[int]]:
    """
    Szegmensek csoportosítása természetes töréspontok alapján.

    Több jelet használ a csoporthatárok meghatározásához:
    - Nagy szünet (gap_threshold) két szegmens között
    - Mondat végi írásjel az aktuális szegmens végén (. ! ?)
    - Maximális csoport időtartam (max_group_duration)
    - Maximális karakter szám (max_group_chars)

    Returns:
        Lista csoportokból, ahol minden csoport a szegmens indexeket tartalmazza.
    """
    if not segments:
        return []

    groups = []
    current_group = [0]
    current_chars = len(segments[0].text_hu)

    for i in range(1, len(segments)):
        prev = segments[i - 1]
        curr = segments[i]

        gap = curr.start - prev.end
        group_duration = curr.end - segments[current_group[0]].start
        ends_with_sentence = bool(re.search(r'[.!?…]\s*$', prev.text_hu))

        should_break = False

        # Hard break: nagy szünet
        if gap >= gap_threshold:
            should_break = True
        # Hard break: maximális időtartam vagy karakterszám túllépése
        elif group_duration > max_group_duration:
            should_break = True
        elif current_chars + len(curr.text_hu) > max_group_chars:
            should_break = True
        # Soft break: közepes szünet + mondatvég
        elif gap >= 0.8 and ends_with_sentence:
            should_break = True

        if should_break:
            groups.append(current_group)
            current_group = [i]
            current_chars = len(curr.text_hu)
        else:
            current_group.append(i)
            current_chars += len(curr.text_hu)

    if current_group:
        groups.append(current_group)

    return groups


def naturalize_segments(
    segments: List[TranslatedSegment],
    translator: str = "openai",
    api_key: str = "",
    model: str = "gpt-4o",
    ollama_url: str = "http://localhost:11434",
    max_retries: int = 3,
    gap_threshold: float = 1.5,
    max_group_duration: float = 60.0,
    max_group_chars: int = 500,
) -> List[NaturalizedGroup]:
    """
    Lefordított szegmensek természetesítése — csoportosítás és LLM átírás.

    1. Szegmensek csoportosítása szünetek és mondathatárok alapján
    2. Csoportonként LLM hívás a természetesebb magyar szövegért
    3. NaturalizedGroup lista visszaadása

    Returns:
        NaturalizedGroup-ok listája a természetesített szöveggel.
    """
    # 1. Csoportosítás
    index_groups = _group_segments_by_gaps(
        segments, gap_threshold, max_group_duration, max_group_chars,
    )
    logger.info(f"Természetesítés: {len(segments)} szegmens → {len(index_groups)} csoport")

    # 2. LLM call_fn előkészítése
    if translator == "ollama":
        import urllib.request
        call_fn = lambda msgs, mdl: _ollama_chat_request(ollama_url, msgs, mdl)
    else:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        call_fn = lambda msgs, mdl: _openai_chat(client, msgs, mdl)

    # 3. Csoportonkénti természetesítés
    result = []
    for g_idx, group_indices in enumerate(index_groups):
        group_segs = [segments[i] for i in group_indices]
        group_start = group_segs[0].start
        group_end = group_segs[-1].end
        target_seconds = group_end - group_start

        # Szövegek összegyűjtése
        en_texts = [seg.text_en for seg in group_segs]
        hu_texts = [seg.text_hu for seg in group_segs]
        combined_en = " ".join(en_texts)
        combined_hu = " ".join(hu_texts)

        # LLM hívás
        system_prompt = NATURALIZE_SYSTEM_PROMPT.format(target_seconds=target_seconds)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                f"Eredeti angol szöveg (referencia):\n{combined_en}\n\n"
                f"Első-körös magyar fordítás (ezt írd át természetesebbre):\n{combined_hu}"
            )},
        ]

        natural_hu = combined_hu  # fallback
        for attempt in range(max_retries):
            try:
                natural_hu = call_fn(messages, model).strip()
                break
            except Exception as e:
                logger.warning(f"Természetesítés hiba, csoport {g_idx+1} (próba {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Természetesítés sikertelen a(z) {g_idx+1}. csoportnál, eredeti szöveg használata")

        logger.info(f"  Csoport {g_idx+1}/{len(index_groups)} [{group_start:.1f}-{group_end:.1f}s] "
                     f"({len(group_segs)} szegmens, {target_seconds:.0f}s)")
        logger.debug(f"    Eredeti: {combined_hu[:80]}...")
        logger.debug(f"    Természetes: {natural_hu[:80]}...")

        result.append(NaturalizedGroup(
            start=group_start,
            end=group_end,
            text_en=combined_en,
            text_hu_original=combined_hu,
            text_hu_natural=natural_hu,
            source_indices=group_indices,
        ))

    logger.info(f"Természetesítés kész: {len(result)} csoport")
    return result


def split_natural_group_to_segments(
    group: NaturalizedGroup,
) -> List[TranslatedSegment]:
    """
    Egy természetesített csoport szétbontása mondatokra TTS-hez.

    A mondatokat arányosan elosztja a csoport időtartamán belül,
    hogy a TTS ne kapjon túl hosszú szövegeket.

    Returns:
        TranslatedSegment-ek listája a csoport időablakán belül elosztva.
    """
    text = group.text_hu_natural.strip()
    if not text:
        return []

    # Mondatokra bontás — pont, felkiáltójel, kérdőjel, három pont után
    sentences = re.split(r'(?<=[.!?…])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        sentences = [text]

    total_duration = group.end - group.start
    total_chars = sum(len(s) for s in sentences)

    if total_chars == 0:
        return [TranslatedSegment(
            start=group.start, end=group.end,
            text_en=group.text_en, text_hu=text,
        )]

    # Arányos időelosztás a mondatok karakter-hossza alapján
    segments = []
    current_start = group.start

    for i, sentence in enumerate(sentences):
        char_ratio = len(sentence) / total_chars
        duration = total_duration * char_ratio
        seg_end = current_start + duration

        # Utolsó mondat: pontosan a csoport végéig
        if i == len(sentences) - 1:
            seg_end = group.end

        segments.append(TranslatedSegment(
            start=current_start,
            end=seg_end,
            text_en=group.text_en if i == 0 else "",
            text_hu=sentence,
        ))
        current_start = seg_end

    return segments
