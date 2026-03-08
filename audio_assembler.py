"""Audio összeállítás modul — TTS szegmensek időzítése és összefűzése."""

import os
import logging
import subprocess
import shutil

from utils import get_audio_duration

logger = logging.getLogger("hu_dub")


def assemble_audio(
    tts_segments: list,
    total_duration: float,
    work_dir: str,
    target_sample_rate: int = 48000,
) -> str:
    """
    TTS szegmensek összefűzése egyetlen hangsávvá, az eredeti időzítéssel.

    Minden szegmenst tempó-igazít, hogy beleférjen az eredeti idősávba,
    majd az idővonalon a megfelelő pozícióba helyezi.

    Args:
        tts_segments: TTSSegment-ek listája (path, target_start, target_end)
        total_duration: Teljes audio hossz másodpercben
        work_dir: Ideiglenes könyvtár
        target_sample_rate: Kimeneti mintavételi frekvencia

    Returns:
        A végső magyar hangsáv WAV fájl útvonala
    """
    logger.info(f"Audio összeállítás: {len(tts_segments)} szegmens, {total_duration:.1f}s")

    # 1. Szegmensek tempó-igazítása
    adjusted_files = []
    for i, seg in enumerate(tts_segments):
        adjusted_path = os.path.join(work_dir, f"adjusted_{i:04d}.wav")
        target_duration = seg.target_end - seg.target_start

        tts_duration = get_audio_duration(seg.path)

        if tts_duration > 0 and target_duration > 0:
            speed = tts_duration / target_duration
            speed = max(0.5, min(2.0, speed))

            # atempo csak 0.5-2.0 közötti értékeket fogad el
            # ha nagyobb tempó kell, láncolni kell
            atempo_filters = _build_atempo_chain(speed)

            subprocess.run([
                "ffmpeg", "-y", "-i", seg.path,
                "-filter:a", atempo_filters,
                "-ar", str(target_sample_rate), "-ac", "2",
                adjusted_path,
            ], check=True, capture_output=True)

            logger.debug(f"  Szegmens {i}: {speed:.2f}x ({tts_duration:.1f}s → {target_duration:.1f}s)")
        else:
            subprocess.run([
                "ffmpeg", "-y", "-i", seg.path,
                "-ar", str(target_sample_rate), "-ac", "2",
                adjusted_path,
            ], check=True, capture_output=True)

        adjusted_files.append({
            "path": adjusted_path,
            "start": seg.target_start,
        })

    # 2. Alap csend generálása a teljes időtartamra
    silence_path = os.path.join(work_dir, "silence.wav")
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i",
        f"anullsrc=r={target_sample_rate}:cl=stereo:d={total_duration}",
        "-t", str(total_duration),
        silence_path,
    ], check=True, capture_output=True)

    # 3. Szegmensek ráhelyezése az idővonalon
    current_base = silence_path
    for i, seg in enumerate(adjusted_files):
        output_path = os.path.join(work_dir, f"mix_{i:04d}.wav")
        delay_ms = int(seg["start"] * 1000)

        subprocess.run([
            "ffmpeg", "-y",
            "-i", current_base,
            "-i", seg["path"],
            "-filter_complex",
            f"[1:a]adelay={delay_ms}|{delay_ms}[delayed];"
            f"[0:a][delayed]amix=inputs=2:duration=first:normalize=0",
            "-ar", str(target_sample_rate), "-ac", "2",
            output_path,
        ], check=True, capture_output=True)

        current_base = output_path
        logger.debug(f"  Mixelve: szegmens {i} @ {seg['start']:.1f}s")

    # 4. Végső fájl
    final_audio = os.path.join(work_dir, "hungarian_audio.wav")
    shutil.copy2(current_base, final_audio)

    logger.info(f"Magyar hangsáv kész: {final_audio}")
    return final_audio


def _build_atempo_chain(speed: float) -> str:
    """
    Atempo filter lánc generálása.
    Az ffmpeg atempo csak 0.5-2.0 közötti értékeket fogad,
    szükség esetén láncolni kell.
    """
    if 0.5 <= speed <= 2.0:
        return f"atempo={speed}"

    # Láncolás szükséges
    filters = []
    remaining = speed
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining}")
    return ",".join(filters)
