"""Pipeline vezérlő — egy MP4 fájl teljes feldolgozási folyamata."""

import os
import sys
import logging
import asyncio
import tempfile
import shutil

from utils import extract_audio, get_audio_duration, ensure_dir
from transcriber import transcribe
from translator import translate_segments
from voice_cloner import extract_voice_sample, generate_cloned_speech, generate_edge_tts
from audio_assembler import assemble_audio
from subtitle_generator import generate_srt_files, generate_transcript_files
from video_merger import merge_video

logger = logging.getLogger("hu_dub")


class Pipeline:
    """Egy MP4 fájl teljes magyar szinkron feldolgozása."""

    def __init__(
        self,
        input_mp4: str,
        output_dir: str,
        mode: str = "dub",
        whisper_model: str = "medium",
        language: str = None,
        translator: str = "openai",
        openai_api_key: str = "",
        openai_model: str = "gpt-4o",
        ollama_model: str = "gemma3:27b",
        ollama_url: str = "http://localhost:11434",
        tts_method: str = "clone",
        voice_sample_sec: int = 30,
        keep_temp: bool = False,
    ):
        self.input_file = os.path.abspath(input_mp4)
        self.output_dir = os.path.abspath(output_dir)
        self.mode = mode
        self.whisper_model = whisper_model
        self.language = language
        self.translator = translator
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.tts_method = tts_method
        self.voice_sample_sec = voice_sample_sec
        self.keep_temp = keep_temp

        self.base_name = os.path.splitext(os.path.basename(input_mp4))[0]
        self.output_mp4 = os.path.join(output_dir, f"{self.base_name}_HU.mp4")
        self.work_dir = None

    def run(self) -> str:
        self.work_dir = tempfile.mkdtemp(prefix="hu_dub_")
        logger.info(f"Temp könyvtár: {self.work_dir}")

        try:
            ensure_dir(self.output_dir)

            if self.mode == "transcribe":
                return self._run_transcribe()

            # Determine translation model based on backend
            trans_model = self.ollama_model if self.translator == "ollama" else self.openai_model

            # Step count depends on mode
            is_dub = self.mode == "dub"
            total_steps = 7 if is_dub else 4

            # 1. Audio kinyerés
            logger.info("=" * 60)
            logger.info(f"[1/{total_steps}] Audio kinyerés...")
            audio_16k = extract_audio(
                self.input_file,
                os.path.join(self.work_dir, "audio_16k.wav"),
                sample_rate=16000, mono=True,
            )
            total_duration = get_audio_duration(audio_16k)
            logger.info(f"  Audio hossz: {total_duration:.1f}s")

            # 2. Transzkripció
            logger.info("=" * 60)
            logger.info(f"[2/{total_steps}] Transzkripció (Whisper {self.whisper_model})...")
            segments = transcribe(audio_16k, self.whisper_model, language="en")
            if not segments:
                raise RuntimeError("Nem találtam beszédet a videóban!")
            logger.info(f"  {len(segments)} szegmens felismerve")

            # 3. Fordítás
            logger.info("=" * 60)
            logger.info(f"[3/{total_steps}] Fordítás ({self.translator}: {trans_model})...")
            translated = translate_segments(
                segments,
                translator=self.translator,
                api_key=self.openai_api_key,
                model=trans_model,
                ollama_url=self.ollama_url,
            )

            if not is_dub:
                # === SUBTITLE-ONLY MÓD ===
                logger.info("=" * 60)
                logger.info(f"[4/{total_steps}] Feliratok generálása és beágyazás...")

                srt_files = generate_srt_files(translated, self.output_dir, self.base_name)

                # MP4 beágyazott feliratokkal (eredeti audio megtartva, nincs magyar hang)
                self._merge_subtitle_only(srt_files)

                logger.info("=" * 60)
                logger.info(f"✅ Kész (csak feliratok)! {self.output_mp4}")
                return self.output_mp4

            # === DUB MÓD ===
            # 4. Hangminta kinyerés (csak clone módhoz)
            if self.tts_method == "clone":
                logger.info("=" * 60)
                logger.info(f"[4/{total_steps}] Hangminta kinyerés ({self.voice_sample_sec}s)...")
                voice_sample = extract_voice_sample(
                    self.input_file,
                    os.path.join(self.work_dir, "voice_sample.wav"),
                    duration=self.voice_sample_sec,
                )

            # 5. TTS generálás
            logger.info("=" * 60)
            logger.info(f"[5/{total_steps}] Magyar TTS generálás ({self.tts_method})...")
            tts_dir = ensure_dir(os.path.join(self.work_dir, "tts"))

            if self.tts_method == "clone":
                tts_segments = generate_cloned_speech(translated, voice_sample, tts_dir)
            else:
                tts_segments = asyncio.run(generate_edge_tts(translated, tts_dir))

            # 6. Audio összeállítás
            logger.info("=" * 60)
            logger.info(f"[6/{total_steps}] Audio összeállítás...")
            mix_dir = ensure_dir(os.path.join(self.work_dir, "mix"))
            hungarian_audio = assemble_audio(tts_segments, total_duration, mix_dir)

            # 7. Feliratok + Végső MP4
            logger.info("=" * 60)
            logger.info(f"[7/{total_steps}] Feliratok és végső MP4...")
            srt_files = generate_srt_files(translated, self.output_dir, self.base_name)
            result = merge_video(
                self.input_file, hungarian_audio,
                srt_files["en"], srt_files["hu"],
                self.output_mp4,
            )

            logger.info("=" * 60)
            logger.info(f"✅ Kész! {result}")
            return result

        except Exception as e:
            logger.error(f"❌ Hiba: {e}")
            raise

        finally:
            if not self.keep_temp and self.work_dir:
                logger.info(f"Temp könyvtár törlése: {self.work_dir}")
                shutil.rmtree(self.work_dir, ignore_errors=True)
            elif self.work_dir:
                logger.info(f"Temp könyvtár megtartva: {self.work_dir}")

    def _run_transcribe(self) -> str:
        """Transcribe mód: csak transzkripció, SRT + tiszta szöveg kimenet."""
        total_steps = 2

        # 1. Audio kinyerés
        logger.info("=" * 60)
        logger.info(f"[1/{total_steps}] Audio kinyerés...")
        audio_16k = extract_audio(
            self.input_file,
            os.path.join(self.work_dir, "audio_16k.wav"),
            sample_rate=16000, mono=True,
        )
        total_duration = get_audio_duration(audio_16k)
        logger.info(f"  Audio hossz: {total_duration:.1f}s")

        # 2. Transzkripció
        logger.info("=" * 60)
        logger.info(f"[2/{total_steps}] Transzkripció (Whisper {self.whisper_model})...")
        segments = transcribe(audio_16k, self.whisper_model, language=self.language)
        if not segments:
            raise RuntimeError("Nem találtam beszédet a fájlban!")
        logger.info(f"  {len(segments)} szegmens felismerve")

        # Kimeneti fájlok generálása
        ensure_dir(self.output_dir)
        result = generate_transcript_files(segments, self.output_dir, self.base_name)

        logger.info("=" * 60)
        logger.info(f"✅ Transzkripció kész!")
        logger.info(f"  SRT: {result['srt']}")
        logger.info(f"  TXT: {result['txt']}")
        return result["srt"]

    def _merge_subtitle_only(self, srt_files: dict):
        """MP4 feliratokkal de extra audió sáv nélkül."""
        import subprocess

        cmd = [
            "ffmpeg", "-y",
            "-i", self.input_file,
            "-i", srt_files["en"],
            "-i", srt_files["hu"],
            "-map", "0:v:0",
            "-map", "0:a:0",
            "-map", "1:0",
            "-map", "2:0",
            "-c:v", "copy",
            "-c:a", "copy",
            "-c:s", "mov_text",
            "-metadata:s:a:0", "language=eng",
            "-metadata:s:s:0", "language=eng",
            "-metadata:s:s:0", "title=English",
            "-metadata:s:s:1", "language=hun",
            "-metadata:s:s:1", "title=Magyar",
            "-disposition:s:0", "0",
            "-disposition:s:1", "0",
            self.output_mp4,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg hiba: {result.stderr[-500:]}")

        size_mb = os.path.getsize(self.output_mp4) / (1024 * 1024)
        logger.info(f"  Kimeneti fájl: {self.output_mp4} ({size_mb:.1f} MB)")
