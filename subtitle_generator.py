"""Felirat generálás modul — SRT fájlok létrehozása, beolvasása és transzkript export."""

import os
import re
import logging
from typing import List

from utils import format_timestamp_srt

logger = logging.getLogger("hu_dub")


def generate_srt_files(
    segments: list,
    output_dir: str,
    base_name: str,
) -> dict:
    """
    Angol és magyar SRT felirat fájlok generálása.

    Args:
        segments: TranslatedSegment-ek listája (start, end, text_en, text_hu)
        output_dir: Kimeneti könyvtár
        base_name: Fájlnév alap (kiterjesztés nélkül)

    Returns:
        Dict a generált fájlok útvonalaival: {"en": "...en.srt", "hu": "...hu.srt"}
    """
    en_path = os.path.join(output_dir, f"{base_name}.en.srt")
    hu_path = os.path.join(output_dir, f"{base_name}.hu.srt")

    _write_srt(en_path, segments, lang="en")
    _write_srt(hu_path, segments, lang="hu")

    logger.info(f"Feliratok generálva: {en_path}, {hu_path}")
    return {"en": en_path, "hu": hu_path}


def _write_srt(filepath: str, segments: list, lang: str) -> None:
    """SRT fájl írása."""
    with open(filepath, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start_ts = format_timestamp_srt(seg.start)
            end_ts = format_timestamp_srt(seg.end)
            text = seg.text_en if lang == "en" else seg.text_hu

            f.write(f"{i}\n")
            f.write(f"{start_ts} --> {end_ts}\n")
            f.write(f"{text}\n")
            f.write("\n")

    logger.debug(f"SRT fájl írva: {filepath} ({lang}, {len(segments)} szegmens)")


def generate_transcript_files(
    segments: list,
    output_dir: str,
    base_name: str,
) -> dict:
    """
    Transzkript fájlok generálása: SRT (időbélyeges) + tiszta szöveg (időbélyeg nélkül).

    Args:
        segments: Segment-ek listája (start, end, text)
        output_dir: Kimeneti könyvtár
        base_name: Fájlnév alap (kiterjesztés nélkül)

    Returns:
        Dict a generált fájlok útvonalaival: {"srt": "....srt", "txt": "....txt"}
    """
    srt_path = os.path.join(output_dir, f"{base_name}.srt")
    txt_path = os.path.join(output_dir, f"{base_name}.txt")

    # SRT fájl
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start_ts = format_timestamp_srt(seg.start)
            end_ts = format_timestamp_srt(seg.end)
            f.write(f"{i}\n")
            f.write(f"{start_ts} --> {end_ts}\n")
            f.write(f"{seg.text}\n")
            f.write("\n")

    # Tiszta szöveg fájl (időbélyegek nélkül)
    with open(txt_path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(f"{seg.text}\n")

    logger.info(f"Transzkript fájlok generálva: {srt_path}, {txt_path}")
    return {"srt": srt_path, "txt": txt_path}


def parse_srt_file(srt_path: str) -> list:
    """
    SRT fájl beolvasása szegmensekre.

    Returns:
        List of dicts: [{'start': float, 'end': float, 'text': str}, ...]
    """
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = re.split(r"\n\s*\n", content.strip())
    segments = []

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue

        # Find the timestamp line
        timestamp_line = None
        text_start_idx = None
        for idx, line in enumerate(lines):
            if re.match(r"\d{2}:\d{2}:\d{2},\d{3}\s*-->", line.strip()):
                timestamp_line = line.strip()
                text_start_idx = idx + 1
                break

        if not timestamp_line or text_start_idx is None:
            continue

        match = re.match(
            r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})",
            timestamp_line,
        )
        if not match:
            continue

        start = _parse_srt_timestamp(match.group(1))
        end = _parse_srt_timestamp(match.group(2))
        text = "\n".join(lines[text_start_idx:]).strip()

        if text:
            segments.append({"start": start, "end": end, "text": text})

    logger.info(f"SRT beolvasva: {srt_path} ({len(segments)} szegmens)")
    return segments


def _parse_srt_timestamp(ts: str) -> float:
    """SRT időbélyeg → másodperc (HH:MM:SS,mmm)."""
    match = re.match(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})", ts)
    if not match:
        return 0.0
    h, m, s, ms = (
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3)),
        int(match.group(4)),
    )
    return h * 3600 + m * 60 + s + ms / 1000.0


def load_existing_subtitles(hu_srt_path: str, en_srt_path: str = None) -> list:
    """
    Meglévő SRT fájlok betöltése TranslatedSegment objektumokba.

    Args:
        hu_srt_path: Magyar SRT fájl útvonala
        en_srt_path: Angol SRT fájl útvonala (opcionális)

    Returns:
        TranslatedSegment-ek listája
    """
    from translator import TranslatedSegment

    hu_segments = parse_srt_file(hu_srt_path)
    en_segments = parse_srt_file(en_srt_path) if en_srt_path else None

    result = []
    for i, hu_seg in enumerate(hu_segments):
        text_en = ""
        if en_segments and i < len(en_segments):
            text_en = en_segments[i]["text"]

        result.append(
            TranslatedSegment(
                start=hu_seg["start"],
                end=hu_seg["end"],
                text_en=text_en,
                text_hu=hu_seg["text"],
            )
        )

    return result
