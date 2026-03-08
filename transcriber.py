"""Whisper transzkripció modul — audio felismerés szegmens-szintű időbélyegekkel."""

import os
import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger("hu_dub")

AVAILABLE_MODELS = [
    "tiny", "tiny.en", "base", "base.en",
    "small", "small.en", "medium", "medium.en",
    "large-v1", "large-v2", "large-v3", "large",
    "large-v3-turbo", "turbo",
]


@dataclass
class Segment:
    """Egy felismert szövegszegmens időbélyeggel."""
    start: float
    end: float
    text: str


def transcribe(audio_wav: str, model_name: str = "medium") -> List[Segment]:
    """
    Audio fájl transzkripciója Whisper modellel.

    Args:
        audio_wav: 16kHz mono WAV fájl útvonala
        model_name: Whisper modell neve (default: medium)

    Returns:
        Szegmensek listája időbélyegekkel
    """
    import whisper

    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Ismeretlen Whisper modell: '{model_name}'. "
            f"Elérhető modellek: {', '.join(AVAILABLE_MODELS)}"
        )

    logger.info(f"Whisper modell betöltése: {model_name}")
    model = whisper.load_model(model_name)

    logger.info(f"Transzkripció indítása: {os.path.basename(audio_wav)}")
    result = model.transcribe(audio_wav, language="en", verbose=False)

    segments = []
    for seg in result["segments"]:
        text = seg["text"].strip()
        if text:
            segments.append(Segment(
                start=seg["start"],
                end=seg["end"],
                text=text,
            ))

    logger.info(f"Transzkripció kész: {len(segments)} szegmens")
    for s in segments:
        logger.debug(f"  [{s.start:.1f}-{s.end:.1f}] {s.text}")

    return segments
