"""Coqui XTTS-v2 hangklónozás modul — magyar TTS a beszélő eredeti hangjával."""

import os
import logging
import subprocess
from dataclasses import dataclass
from typing import List

logger = logging.getLogger("hu_dub")


def _patch_torchaudio_load():
    """
    Monkey-patch torchaudio.load → soundfile fallback.
    PyTorch 2.9+ torchaudio torchcodec-et használ, ami Windows-on
    hiányzó FFmpeg shared library-k miatt nem töltődik be.
    """
    import torch
    import torchaudio
    import soundfile as sf

    _original_load = torchaudio.load

    def patched_load(filepath, *args, **kwargs):
        try:
            return _original_load(filepath, *args, **kwargs)
        except (RuntimeError, OSError):
            logger.debug(f"torchaudio.load fallback → soundfile: {filepath}")
            data, samplerate = sf.read(str(filepath), dtype="float32")
            tensor = torch.FloatTensor(data)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)  # (1, samples)
            else:
                tensor = tensor.T  # (channels, samples)
            return tensor, samplerate

    torchaudio.load = patched_load

    # Patch the XTTS module's local load_audio too
    import TTS.tts.models.xtts as xtts_mod

    def patched_load_audio(audiopath, sampling_rate):
        audio, lsr = patched_load(audiopath)
        if audio.size(0) != 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        if lsr != sampling_rate:
            audio = torchaudio.functional.resample(audio, lsr, sampling_rate)
        audio = audio.clamp(-1, 1)
        return audio  # (1, samples) — 2D, as XTTS expects

    xtts_mod.load_audio = patched_load_audio
    logger.info("torchaudio.load monkey-patch alkalmazva (soundfile fallback)")


@dataclass
class TTSSegment:
    """Egy generált TTS szegmens."""
    path: str
    target_start: float
    target_end: float
    text: str


def extract_voice_sample(
    input_mp4: str,
    output_wav: str,
    duration: int = 30,
) -> str:
    """
    Hangminta kinyerése az eredeti audioból a hangklónozáshoz.
    Megpróbálja a beszédben leggazdagabb részt kiválasztani.

    Args:
        input_mp4: Eredeti MP4 fájl
        output_wav: Kimeneti WAV fájl (22050Hz, mono — XTTS követelmény)
        duration: Hangminta hossza másodpercben

    Returns:
        A hangminta fájl útvonala
    """
    from utils import get_audio_duration

    # Teljes audio hossza
    total_duration = get_audio_duration(input_mp4)
    sample_duration = min(duration, total_duration)

    # Az elejétől vesszük a mintát (általában ott van a beszéd)
    # de kihagyjuk az első 0.5 másodpercet (intro zaj)
    start_time = 0.5 if total_duration > duration + 1 else 0

    subprocess.run([
        "ffmpeg", "-y",
        "-i", input_mp4,
        "-ss", str(start_time),
        "-t", str(sample_duration),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "22050",
        "-ac", "1",
        output_wav,
    ], check=True, capture_output=True)

    logger.info(f"Hangminta kinyerve: {sample_duration:.0f}s @ 22050Hz mono")
    return output_wav


def generate_cloned_speech(
    segments: list,
    voice_sample_wav: str,
    output_dir: str,
    device: str = "cuda",
) -> List[TTSSegment]:
    """
    Magyar beszéd generálása XTTS-v2-vel a klónozott hanggal.

    Args:
        segments: TranslatedSegment-ek listája (text_hu kell)
        voice_sample_wav: Hangminta WAV fájl a klónozáshoz
        output_dir: Kimeneti könyvtár a TTS fájlokhoz
        device: "cuda" vagy "cpu"

    Returns:
        Generált TTS szegmensek listája
    """
    import torch

    # Patch torchaudio.load before importing TTS model
    _patch_torchaudio_load()

    from TTS.api import TTS

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA nem elérhető, CPU-ra váltás")
        device = "cpu"

    logger.info(f"XTTS-v2 modell betöltése ({device})...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    tts_segments = []
    total = len(segments)

    for i, seg in enumerate(segments):
        out_path = os.path.join(output_dir, f"tts_seg_{i:04d}.wav")

        logger.info(f"  TTS [{i + 1}/{total}] [{seg.start:.1f}-{seg.end:.1f}] {seg.text_hu[:60]}...")

        try:
            tts.tts_to_file(
                text=seg.text_hu,
                speaker_wav=voice_sample_wav,
                language="hu",
                file_path=out_path,
            )
        except Exception as e:
            logger.error(f"  TTS hiba a(z) {i}. szegmensnél: {e}")
            # Üres audio generálása fallback-ként
            duration = seg.end - seg.start
            subprocess.run([
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", f"anullsrc=r=22050:cl=mono:d={duration}",
                "-t", str(duration), out_path,
            ], check=True, capture_output=True)

        tts_segments.append(TTSSegment(
            path=out_path,
            target_start=seg.start,
            target_end=seg.end,
            text=seg.text_hu,
        ))

    logger.info(f"TTS generálás kész: {len(tts_segments)} szegmens")
    return tts_segments


async def generate_edge_tts(
    segments: list,
    output_dir: str,
    voice: str = "hu-HU-TamasNeural",
) -> List[TTSSegment]:
    """
    Magyar beszéd generálása Edge-TTS-sel (általános magyar hang, nincs klónozás).

    Elérhető magyar hangok:
    - hu-HU-TamasNeural (férfi, alapértelmezett)
    - hu-HU-NoemiNeural (női)

    Args:
        segments: TranslatedSegment-ek listája
        output_dir: Kimeneti könyvtár
        voice: Edge-TTS hang neve

    Returns:
        Generált TTS szegmensek listája
    """
    import edge_tts

    logger.info(f"Edge-TTS generálás ({voice})...")
    tts_segments = []
    total = len(segments)

    for i, seg in enumerate(segments):
        out_path = os.path.join(output_dir, f"tts_seg_{i:04d}.mp3")

        logger.info(f"  TTS [{i + 1}/{total}] [{seg.start:.1f}-{seg.end:.1f}] {seg.text_hu[:60]}...")

        try:
            communicate = edge_tts.Communicate(seg.text_hu, voice)
            await communicate.save(out_path)
        except Exception as e:
            logger.error(f"  Edge-TTS hiba a(z) {i}. szegmensnél: {e}")
            duration = seg.end - seg.start
            subprocess.run([
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", f"anullsrc=r=22050:cl=mono:d={duration}",
                "-t", str(duration), out_path,
            ], check=True, capture_output=True)

        tts_segments.append(TTSSegment(
            path=out_path,
            target_start=seg.start,
            target_end=seg.end,
            text=seg.text_hu,
        ))

    logger.info(f"Edge-TTS generálás kész: {len(tts_segments)} szegmens")
    return tts_segments
