"""FFmpeg videó összeállítás modul — végső MP4 generálás minden sávval."""

import os
import logging
import subprocess

logger = logging.getLogger("hu_dub")


def merge_video(
    input_mp4: str,
    hungarian_audio: str,
    output_mp4: str,
) -> str:
    """
    Végső MP4 összeállítása:
    - Videó stream: eredeti (nem kódoljuk újra)
    - Audio 1: eredeti angol
    - Audio 2: magyar szinkron

    Args:
        input_mp4: Eredeti MP4 fájl
        hungarian_audio: Magyar hangsáv WAV
        output_mp4: Kimeneti MP4 fájl

    Returns:
        Kimeneti fájl útvonala
    """
    logger.info("Végső MP4 összeállítása...")

    cmd = [
        "ffmpeg", "-y",
        "-i", input_mp4,           # input 0: eredeti videó + audio
        "-i", hungarian_audio,     # input 1: magyar audio
        # Stream mapping
        "-map", "0:v:0",           # videó az eredetiből
        "-map", "0:a:0",           # eredeti angol audio
        "-map", "1:a:0",           # magyar szinkron audio
        # Codec beállítások
        "-c:v", "copy",            # videó: nem kódoljuk újra
        "-c:a:0", "aac", "-b:a:0", "192k",  # angol audio: AAC
        "-c:a:1", "aac", "-b:a:1", "192k",  # magyar audio: AAC
        # Metaadatok — nyelvi címkék
        "-metadata:s:a:0", "language=eng",
        "-metadata:s:a:0", "title=English",
        "-metadata:s:a:1", "language=hun",
        "-metadata:s:a:1", "title=Magyar",
        # Alapértelmezett sávok
        "-disposition:a:0", "default",
        "-disposition:a:1", "0",
        "-shortest",
        output_mp4,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"FFmpeg hiba:\n{result.stderr}")
        raise RuntimeError(f"FFmpeg hiba: {result.stderr[-500:]}")

    size_mb = os.path.getsize(output_mp4) / (1024 * 1024)
    logger.info(f"Kimeneti fájl: {output_mp4} ({size_mb:.1f} MB)")

    # Ellenőrzés: sávok megjelenítése
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", output_mp4],
        capture_output=True, text=True,
    )
    stream_types = []
    for line in probe.stdout.split("\n"):
        if "codec_type=" in line:
            stream_types.append(line.split("=")[1])
    logger.info(f"Sávok: {', '.join(stream_types)}")

    return output_mp4
