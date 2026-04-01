"""Felirat generálás modul — SRT fájlok létrehozása és transzkript export."""

import os
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
        Dict a generált fájlok útvonalaival: {"en": "..._EN.srt", "hu": "..._HU.srt"}
    """
    en_path = os.path.join(output_dir, f"{base_name}_EN.srt")
    hu_path = os.path.join(output_dir, f"{base_name}_HU.srt")

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
