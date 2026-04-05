"""Pipeline vezérlő — egy MP4 fájl teljes feldolgozási folyamata."""

import os
import sys
import logging
import asyncio
import tempfile
import shutil

from utils import extract_audio, get_audio_duration, ensure_dir
from transcriber import transcribe
from translator import translate_segments, naturalize_segments, split_natural_group_to_segments
from voice_cloner import extract_voice_sample, generate_cloned_speech, generate_edge_tts
from audio_assembler import assemble_audio
from subtitle_generator import generate_srt_files, generate_transcript_files, load_existing_subtitles, generate_natural_srt
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
        dub_style: str = "precise",
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
        self.dub_style = dub_style

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

            trans_model = self.ollama_model if self.translator == "ollama" else self.openai_model
            is_dub = self.mode == "dub"

            # Check for existing Hungarian subtitle next to input file
            existing_hu = self._find_existing_srt("hu")
            existing_en = self._find_existing_srt("en")

            if existing_hu:
                # === EXISTING SUBTITLE — skip transcription + translation ===
                translated = load_existing_subtitles(existing_hu, existing_en)
                logger.info(f"Meglévő magyar felirat betöltve: {existing_hu} ({len(translated)} szegmens)")

                if not is_dub:
                    if self.dub_style == "natural":
                        # Subtitle mód + natural: természetesítés
                        total_steps = 2
                        step = 0

                        step += 1
                        logger.info("=" * 60)
                        logger.info(f"[{step}/{total_steps}] Természetesítés ({self.translator}: {trans_model})...")
                        natural_groups = naturalize_segments(
                            translated,
                            translator=self.translator,
                            api_key=self.openai_api_key,
                            model=trans_model,
                            ollama_url=self.ollama_url,
                        )

                        step += 1
                        logger.info("=" * 60)
                        logger.info(f"[{step}/{total_steps}] Természetesített felirat generálása...")
                        natural_srt = generate_natural_srt(natural_groups, self.output_dir, self.base_name)

                        logger.info("=" * 60)
                        logger.info(f"✅ Természetesített felirat kész! {natural_srt}")
                        return natural_srt
                    else:
                        logger.info("=" * 60)
                        logger.info("✅ Feliratok már léteznek!")
                        return existing_hu

                # Dub mode: still need audio duration
                natural_extra = 1 if self.dub_style == "natural" else 0
                total_steps = (5 if self.tts_method == "clone" else 4) + natural_extra
                step = 0

                step += 1
                logger.info("=" * 60)
                logger.info(f"[{step}/{total_steps}] Audio kinyerés (időtartam)...")
                audio_16k = extract_audio(
                    self.input_file,
                    os.path.join(self.work_dir, "audio_16k.wav"),
                    sample_rate=16000, mono=True,
                )
                total_duration = get_audio_duration(audio_16k)
                logger.info(f"  Audio hossz: {total_duration:.1f}s")

            else:
                # === NO EXISTING SUBTITLE — full transcription + translation ===
                natural_extra = 1 if self.dub_style == "natural" else 0
                if is_dub:
                    total_steps = (8 if self.tts_method == "clone" else 7) + natural_extra
                else:
                    total_steps = 4 + natural_extra
                step = 0

                step += 1
                logger.info("=" * 60)
                logger.info(f"[{step}/{total_steps}] Audio kinyerés...")
                audio_16k = extract_audio(
                    self.input_file,
                    os.path.join(self.work_dir, "audio_16k.wav"),
                    sample_rate=16000, mono=True,
                )
                total_duration = get_audio_duration(audio_16k)
                logger.info(f"  Audio hossz: {total_duration:.1f}s")

                step += 1
                logger.info("=" * 60)
                logger.info(f"[{step}/{total_steps}] Transzkripció (Whisper {self.whisper_model})...")
                segments = transcribe(audio_16k, self.whisper_model, language="en")
                if not segments:
                    raise RuntimeError("Nem találtam beszédet a videóban!")
                logger.info(f"  {len(segments)} szegmens felismerve")

                step += 1
                logger.info("=" * 60)
                logger.info(f"[{step}/{total_steps}] Fordítás ({self.translator}: {trans_model})...")
                translated = translate_segments(
                    segments,
                    translator=self.translator,
                    api_key=self.openai_api_key,
                    model=trans_model,
                    ollama_url=self.ollama_url,
                )

                step += 1
                logger.info("=" * 60)
                logger.info(f"[{step}/{total_steps}] Feliratok generálása...")
                generate_srt_files(translated, self.output_dir, self.base_name)

                if not is_dub:
                    if self.dub_style == "natural":
                        step += 1
                        logger.info("=" * 60)
                        logger.info(f"[{step}/{total_steps}] Természetesítés ({self.translator}: {trans_model})...")
                        natural_groups = naturalize_segments(
                            translated,
                            translator=self.translator,
                            api_key=self.openai_api_key,
                            model=trans_model,
                            ollama_url=self.ollama_url,
                        )
                        generate_natural_srt(natural_groups, self.output_dir, self.base_name)

                    hu_srt = os.path.join(self.output_dir, f"{self.base_name}.hu.srt")
                    logger.info("=" * 60)
                    logger.info(f"✅ Feliratok generálva! {hu_srt}")
                    return hu_srt

            # === DUB MODE — TTS + audio assembly + final MP4 ===

            # Természetesítés (natural mód)
            if self.dub_style == "natural":
                step += 1
                logger.info("=" * 60)
                logger.info(f"[{step}/{total_steps}] Természetesítés ({self.translator}: {trans_model})...")
                natural_groups = naturalize_segments(
                    translated,
                    translator=self.translator,
                    api_key=self.openai_api_key,
                    model=trans_model,
                    ollama_url=self.ollama_url,
                )
                generate_natural_srt(natural_groups, self.output_dir, self.base_name)

                # Természetesített csoportok mondatokra bontása TTS-hez
                tts_input_segments = []
                for group in natural_groups:
                    tts_input_segments.extend(split_natural_group_to_segments(group))
                logger.info(f"  Természetesített szegmensek TTS-hez: {len(tts_input_segments)}")
            else:
                tts_input_segments = translated

            if self.tts_method == "clone":
                step += 1
                logger.info("=" * 60)
                logger.info(f"[{step}/{total_steps}] Hangminta kinyerés ({self.voice_sample_sec}s)...")
                voice_sample = extract_voice_sample(
                    self.input_file,
                    os.path.join(self.work_dir, "voice_sample.wav"),
                    duration=self.voice_sample_sec,
                )

            step += 1
            logger.info("=" * 60)
            logger.info(f"[{step}/{total_steps}] Magyar TTS generálás ({self.tts_method})...")
            tts_dir = ensure_dir(os.path.join(self.work_dir, "tts"))

            if self.tts_method == "clone":
                tts_segments = generate_cloned_speech(tts_input_segments, voice_sample, tts_dir)
            else:
                tts_segments = asyncio.run(generate_edge_tts(tts_input_segments, tts_dir))

            step += 1
            logger.info("=" * 60)
            logger.info(f"[{step}/{total_steps}] Audio összeállítás...")
            mix_dir = ensure_dir(os.path.join(self.work_dir, "mix"))
            hungarian_audio = assemble_audio(tts_segments, total_duration, mix_dir)

            step += 1
            logger.info("=" * 60)
            logger.info(f"[{step}/{total_steps}] Végső MP4...")
            result = merge_video(
                self.input_file, hungarian_audio,
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

    def _find_existing_srt(self, lang_suffix: str) -> str:
        """Meglévő SRT felirat keresése a bemeneti fájl mellett (és az output könyvtárban)."""
        input_dir = os.path.dirname(self.input_file)
        srt_path = os.path.join(input_dir, f"{self.base_name}.{lang_suffix}.srt")
        if os.path.isfile(srt_path):
            return srt_path
        # Output dir is different — check there too
        if os.path.abspath(self.output_dir) != os.path.abspath(input_dir):
            srt_path = os.path.join(self.output_dir, f"{self.base_name}.{lang_suffix}.srt")
            if os.path.isfile(srt_path):
                return srt_path
        return None
