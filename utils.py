"""Közös segédfüggvények a hu_dub alkalmazáshoz."""

import os
import subprocess
import logging

logger = logging.getLogger("hu_dub")


def get_audio_duration(filepath: str) -> float:
    """Audio fájl hossza másodpercben."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", filepath],
        capture_output=True, text=True, check=True
    )
    return float(result.stdout.strip())


def extract_audio(input_mp4: str, output_wav: str, sample_rate: int = 16000, mono: bool = True) -> str:
    """Audio kinyerése média fájlból (MP4/MP3) WAV formátumba."""
    cmd = [
        "ffmpeg", "-y", "-i", input_mp4,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
    ]
    if mono:
        cmd.extend(["-ac", "1"])
    cmd.append(output_wav)
    subprocess.run(cmd, check=True, capture_output=True)
    logger.info(f"Audio kinyerve: {output_wav}")
    return output_wav


def ensure_dir(path: str) -> str:
    """Könyvtár létrehozása ha nem létezik."""
    os.makedirs(path, exist_ok=True)
    return path


def format_timestamp_srt(seconds: float) -> str:
    """Másodperc → SRT időbélyeg formátum (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
