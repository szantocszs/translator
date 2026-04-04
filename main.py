#!/usr/bin/env python3
"""
hu_dub — Magyar szinkron generáló CLI alkalmazás

Módok:
  transcribe — Csak transzkripció: audio/videó fájlból SRT + tiszta szöveg (magyar nyelv is)
  subtitle   — Felirat: transzkripció + fordítás + külső SRT fájlok
  dub        — Teljes szinkron: felirat + magyar hangsáv (hangklónozással vagy Edge-TTS-sel)

Ha a bemeneti fájl mellett már létezik .hu.srt felirat, a subtitle/dub mód
azt használja a transzkripció + fordítás helyett.

TTS módszerek (dub módhoz):
  clone     — XTTS-v2 hangklónozás az eredeti beszélő hangjával
  edge      — Edge-TTS általános magyar hang (gyorsabb, nincs GPU igény)

Fordítás backend-ek:
  openai    — OpenAI GPT API (fizetős, legjobb minőség)
  ollama    — Helyi Ollama LLM (ingyenes, GPU-n fut)
"""

import argparse
import logging
import os
import sys
import glob as glob_module

from transcriber import AVAILABLE_MODELS
from translator import RECOMMENDED_OLLAMA_MODELS
from pipeline import Pipeline

SUPPORTED_EXTENSIONS = (".mp4", ".mp3")


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger = logging.getLogger("hu_dub")
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


def get_openai_key(args_key: str = None) -> str:
    """OpenAI API kulcs: arg > AI__OpenAiKey env > OPENAI_API_KEY env."""
    if args_key:
        return args_key
    key = os.environ.get("AI__OpenAiKey", "")
    if key:
        return key
    key = os.environ.get("OPENAI_API_KEY", "")
    if key:
        return key
    return ""


def find_media_files(path: str, batch: bool, mode: str, recursive: bool = False) -> list:
    """Bemeneti fájl(ok) keresése. Transcribe mód mp3-at és mp4-et is elfogad."""
    if mode == "transcribe":
        valid_exts = SUPPORTED_EXTENSIONS
    else:
        valid_exts = (".mp4",)

    if os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        if ext not in valid_exts:
            exts_str = ", ".join(valid_exts)
            print(f"Hiba: Nem támogatott fájlformátum: {path} (elfogadott: {exts_str})", file=sys.stderr)
            sys.exit(1)
        return [path]

    if os.path.isdir(path):
        if not batch:
            print(f"Hiba: '{path}' könyvtár. Használd a --batch opciót.", file=sys.stderr)
            sys.exit(1)
        files = []
        if recursive:
            for ext in valid_exts:
                files.extend(glob_module.glob(os.path.join(path, "**", f"*{ext}"), recursive=True))
        else:
            for ext in valid_exts:
                files.extend(glob_module.glob(os.path.join(path, f"*{ext}")))
        files = sorted(set(files))
        files = [f for f in files if not f.endswith("_HU.mp4")]
        if not files:
            exts_str = ", ".join(valid_exts)
            recursive_hint = " (rekurzív)" if recursive else ""
            print(f"Hiba: Nem találtam média fájlokat ({exts_str}){recursive_hint}: {path}", file=sys.stderr)
            sys.exit(1)
        return files

    print(f"Hiba: Nem létezik: {path}", file=sys.stderr)
    sys.exit(1)


def main():
    ollama_models_help = ", ".join(RECOMMENDED_OLLAMA_MODELS)

    parser = argparse.ArgumentParser(
        prog="hu_dub",
        description="Magyar szinkron generáló — MP4 videókhoz és audio transzkripció",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Példák:
  # Transzkripció (mp3/mp4 → SRT + tiszta szöveg, automatikus nyelvfelismerés)
  python hu_dub/main.py -i "podcast.mp3" --mode transcribe

  # Magyar nyelvű transzkripció (large-v3 modell ajánlott)
  python hu_dub/main.py -i "eloadas.mp4" --mode transcribe --language hu -w large-v3

  # Csak feliratok (nincs szinkron hang)
  python hu_dub/main.py -i "video.mp4" --mode subtitle

  # Szinkron Edge-TTS-sel (gyors, általános magyar hang)
  python hu_dub/main.py -i "video.mp4" --mode dub --tts-method edge

  # Szinkron hangklónozással (XTTS-v2, a beszélő hangjával)
  python hu_dub/main.py -i "video.mp4" --mode dub --tts-method clone

  # Ollama-val fordítás (helyi LLM, ingyenes)
  python hu_dub/main.py -i "video.mp4" --translator ollama --ollama-model gemma3:27b

  # Batch feldolgozás
  python hu_dub/main.py -i ./videos/ --batch -w large-v3 --mode dub --tts-method edge

  # Batch feldolgozás rekurzív könyvtár bejárással
  python hu_dub/main.py -i ./videos/ --batch -r --mode transcribe --language hu

Ajánlott Ollama modellek magyar fordításhoz (32GB VRAM):
  {ollama_models_help}
        """,
    )

    # Alap paraméterek
    parser.add_argument("-i", "--input", required=True, help="Bemeneti MP4/MP3 fájl vagy könyvtár")
    parser.add_argument("-o", "--output", help="Kimeneti könyvtár (default: bemeneti mappa)")
    parser.add_argument("--batch", action="store_true", help="Könyvtár mód: minden fájl feldolgozása")
    parser.add_argument("-r", "--recursive", action="store_true",
                        help="Rekurzív könyvtár bejárás (--batch opcióval együtt)")
    parser.add_argument("--skip-existing", action="store_true", help="Már létező kimeneti fájlok kihagyása")
    parser.add_argument("--keep-temp", action="store_true", help="Temp fájlok megtartása (debug)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Részletes log")

    # Mód
    parser.add_argument(
        "--mode", choices=["transcribe", "subtitle", "dub"], default="dub",
        help="Feldolgozási mód: transcribe (csak átírás) | subtitle (felirat) | dub (szinkron) [default: dub]",
    )

    # Whisper
    parser.add_argument(
        "-w", "--whisper-model", default=None, choices=AVAILABLE_MODELS,
        help="Whisper modell [default: large-v3 transcribe módban, medium egyébként]",
    )

    # Nyelv (transzkripció)
    parser.add_argument(
        "--language", default=None,
        help="Audio nyelve a transzkripcióhoz (pl. hu, en, de). "
             "Nincs megadva = automatikus felismerés. "
             "subtitle/dub módban automatikusan 'en'.",
    )

    # Fordítás
    parser.add_argument(
        "--translator", choices=["openai", "ollama"], default="openai",
        help="Fordítás backend: openai (GPT API) | ollama (helyi LLM) [default: openai]",
    )
    parser.add_argument(
        "--openai-api-key", help="OpenAI API kulcs (default: AI__OpenAiKey / OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--openai-model", default="gpt-4o", help="OpenAI modell [default: gpt-4o]",
    )
    parser.add_argument(
        "--ollama-model", default="gemma3:27b",
        help=f"Ollama modell [default: gemma3:27b]. Ajánlott: {ollama_models_help}",
    )
    parser.add_argument(
        "--ollama-url", default="http://localhost:11434", help="Ollama szerver URL [default: http://localhost:11434]",
    )

    # TTS (dub módhoz)
    parser.add_argument(
        "--tts-method", choices=["clone", "edge"], default="clone",
        help="TTS módszer: clone (XTTS-v2 hangklón) | edge (Edge-TTS általános hang) [default: clone]",
    )
    parser.add_argument(
        "--voice-sample-sec", type=int, default=30,
        help="Hangminta hossza a klónozáshoz [default: 30s, csak clone módhoz]",
    )

    args = parser.parse_args()
    logger = setup_logging(args.verbose)

    # Whisper modell default: transcribe módnál large-v3, egyébként medium
    if args.whisper_model is None:
        args.whisper_model = "large-v3" if args.mode == "transcribe" else "medium"

    # Nyelv kezelés: subtitle/dub módban fix "en", transcribe módban user-defined vagy None (auto)
    language = args.language
    if args.mode in ("subtitle", "dub"):
        language = "en"

    # Validáció: OpenAI kulcs csak subtitle/dub módban kell (ha openai translator)
    api_key = ""
    if args.mode != "transcribe" and args.translator == "openai":
        api_key = get_openai_key(args.openai_api_key)
        if not api_key:
            print(
                "Hiba: Nincs OpenAI API kulcs.\n"
                "Megadási módok: --openai-api-key, AI__OpenAiKey env var, OPENAI_API_KEY env var\n"
                "Vagy használj Ollama-t: --translator ollama",
                file=sys.stderr,
            )
            sys.exit(1)

    if args.recursive and not args.batch:
        print("Hiba: --recursive csak --batch opcióval használható.", file=sys.stderr)
        sys.exit(1)

    media_files = find_media_files(args.input, args.batch, args.mode, args.recursive)

    if args.output:
        output_dir = args.output
    elif os.path.isfile(args.input):
        output_dir = os.path.dirname(os.path.abspath(args.input))
    else:
        output_dir = os.path.abspath(args.input)

    # Info kiírás
    logger.info("hu_dub — Magyar szinkron generáló")
    logger.info(f"  Mód: {args.mode}")
    logger.info(f"  Fájlok: {len(media_files)} db")
    logger.info(f"  Whisper: {args.whisper_model}")
    if args.mode == "transcribe":
        lang_display = language if language else "automatikus felismerés"
        logger.info(f"  Nyelv: {lang_display}")
    else:
        trans_info = f"{args.translator} ({args.openai_model})" if args.translator == "openai" else f"{args.translator} ({args.ollama_model})"
        logger.info(f"  Fordítás: {trans_info}")
    if args.mode == "dub":
        logger.info(f"  TTS: {args.tts_method}" + (f" (hangminta: {args.voice_sample_sec}s)" if args.tts_method == "clone" else ""))
    if args.recursive:
        logger.info(f"  Rekurzív: igen")
    logger.info(f"  Kimenet: {output_dir}" + (" (forrásfájl mellé)" if args.recursive and not args.output else ""))

    success = 0
    failed = 0
    skipped = 0

    for i, media_file in enumerate(media_files):
        base = os.path.splitext(os.path.basename(media_file))[0]

        # Rekurzív mód + nincs explicit -o: kimenet a forrásfájl könyvtárába
        if args.recursive and not args.output:
            file_output_dir = os.path.dirname(os.path.abspath(media_file))
        else:
            file_output_dir = output_dir

        if args.mode == "transcribe":
            out_check = os.path.join(file_output_dir, f"{base}.srt")
        elif args.mode == "subtitle":
            out_check = os.path.join(file_output_dir, f"{base}.hu.srt")
        else:
            out_check = os.path.join(file_output_dir, f"{base}_HU.mp4")

        logger.info(f"\n{'='*60}")
        logger.info(f"[{i+1}/{len(media_files)}] {os.path.basename(media_file)}")
        if args.recursive:
            logger.info(f"  Forrás: {media_file}")
            logger.info(f"  Kimenet: {file_output_dir}")
        logger.info(f"{'='*60}")

        if args.skip_existing and os.path.exists(out_check):
            logger.info(f"  Kihagyva (már létezik): {out_check}")
            skipped += 1
            continue

        try:
            pipeline = Pipeline(
                input_mp4=media_file,
                output_dir=file_output_dir,
                mode=args.mode,
                whisper_model=args.whisper_model,
                language=language,
                translator=args.translator,
                openai_api_key=api_key,
                openai_model=args.openai_model,
                ollama_model=args.ollama_model,
                ollama_url=args.ollama_url,
                tts_method=args.tts_method,
                voice_sample_sec=args.voice_sample_sec,
                keep_temp=args.keep_temp,
            )
            pipeline.run()
            success += 1
        except Exception as e:
            logger.error(f"  ❌ Feldolgozás sikertelen: {e}")
            failed += 1
            if not args.batch:
                sys.exit(1)

    logger.info(f"\n{'='*60}")
    logger.info(f"Összesítés: {success} sikeres, {failed} sikertelen, {skipped} kihagyva")
    logger.info(f"{'='*60}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
